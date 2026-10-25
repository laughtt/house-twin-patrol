"""The duck's twin home for tests and missions: the MuJoCo scene of a house (python -m twin sim), a duck-height eye that
renders, Pollen's 8x8 ToF model, a stand-in for the object detector, and the duck's own motion (it stops at walls and
furniture; its odometry keeps counting, with the drift of a real walking duck).

The stand-in detector: the true landmarks in view and not hidden, 70 % hit rate, 3° bearing noise, 10 % wrong class,
5 % false sightings per frame, range for 60 % of hits. The plain boxes of the twin can't be recognised by YOLOX, so the
real detector is tested on real photos (eval/lowview_test.py), not here.
"""
import importlib.util
import json
import math
import os
from pathlib import Path

import cv2 as cv
import mujoco
import numpy as np

from agent.localize import CELL, HouseMap

ROOT = Path(__file__).resolve().parents[1]
ROBOT = Path.home() / "Pollen/microduck_rl/src/mjlab_microduck/robot/microduck"
EYE_H, FOVY, W, H = 0.2, 57.7, 480, 640
RADIUS = 0.1  # the duck's half width: closer than this to a wall or a leg, it bumps


def _tof_module():
    spec = importlib.util.spec_from_file_location("tof", ROBOT.parents[1] / "sim/tof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TwinSim:
    def __init__(self, house_dir=ROOT / "out/current", seed=0, extra_geoms="", stickers=False):
        house_dir = Path(house_dir)
        self.house = json.loads((house_dir / "house.json").read_text())
        f = house_dir / "sim_frame.json"  # written by `python -m twin sim`; any frame works for a test of its own
        frame = json.loads(f.read_text()) if f.exists() else {"theta": 0.0, "pos": [0.0, 0.0]}
        self.th, (self.px, self.py) = frame["theta"], frame["pos"]
        self.world = HouseMap(self.house)
        k = 2 * int(RADIUS / CELL) + 1
        self.blocked = cv.dilate(self.world.grid.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))).astype(bool)
        self.clearance = cv.distanceTransform((~self.world.grid).astype(np.uint8), cv.DIST_L2, 5) * CELL  # metres to the nearest thing
        self.rng = np.random.default_rng(seed)
        from twin.to_mjcf import house_xml  # this house's own scene, so several houses can run side by side
        tag = f"{os.getpid()}_{seed}"
        (ROBOT / f"twin_house_{tag}.xml").write_text(house_xml(self.house if stickers else {**self.house, "markers": []}, frame)[0])
        xml = ROBOT / f"twin_sim_{tag}.xml"
        xml.write_text(f'''<mujoco model="twin_sim">
  <include file="twin_house_{tag}.xml"/>
  <visual><global offwidth="800" offheight="800"/><headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/></visual>
  <worldbody>
    <light pos="0 0 2.9" dir="0 0 -1" directional="true" castshadow="false"/>
    <body name="eye" mocap="true" pos="0 0 {EYE_H}">
      <camera name="eye" fovy="{FOVY}" xyaxes="0 -1 0 0 0 1"/>
      <site name="tof" size="0.01"/>
    </body>
    {extra_geoms}
  </worldbody>
</mujoco>''')
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        xml.unlink()
        (ROBOT / f"twin_house_{tag}.xml").unlink()
        self.data = mujoco.MjData(self.model)
        self._tof = _tof_module().Tof(self.model, self.model.site("tof").id, seed=seed)
        self._renderer = None
        fy = H / 2 / math.tan(math.radians(FOVY) / 2)
        self.K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
        self.hfov = 2 * math.atan(W / 2 / fy)
        self.classes = sorted({lm["cls"] for lm in self.world.landmarks})

    def to_world(self, x, y):
        """House frame (house.json) → MuJoCo world (the duck spawns at the origin facing +x)."""
        c, s = math.cos(self.th), math.sin(self.th)
        return c * x - s * y + self.px, s * x + c * y + self.py

    def _place(self, x, y, yaw):
        wx, wy = self.to_world(x, y)
        self.data.mocap_pos[0] = [wx, wy, EYE_H]
        self.data.mocap_quat[0] = [math.cos((yaw + self.th) / 2), 0, 0, math.sin((yaw + self.th) / 2)]
        mujoco.mj_forward(self.model, self.data)

    def tof(self, x, y, yaw):
        """(distances mm, status) of the 8x8 sensor, as Pollen's simulator reports them."""
        self._place(x, y, yaw)
        return self._tof.frame(self.data)

    def image(self, x, y, yaw):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, H, W)
        self._place(x, y, yaw)
        self._renderer.update_scene(self.data, camera="eye")
        return self._renderer.render()

    def room_colours(self, n=24):
        """Each room's colour fingerprint (perceive.colour_hist) as the twin's camera sees it, from n views. The scan's
        come from real photos, which the twin's plain floors and walls don't look like: on its renders they pick the
        right room 16 % of the time, these 66 %."""
        from agent.perceive import colour_hist
        cells = np.argwhere(~self.blocked & (self.world.room_of >= 0))
        out = []
        for k in range(len(self.world.rooms)):
            mine = cells[self.world.room_of[cells[:, 0], cells[:, 1]] == k]
            views = [((j - 0.5) * CELL + self.world.o[0], (i - 0.5) * CELL + self.world.o[1], self.rng.uniform(-math.pi, math.pi))
                     for i, j in mine[self.rng.integers(len(mine), size=n)]] if len(mine) else []
            out.append(np.mean([colour_hist(cv.cvtColor(self.image(*v), cv.COLOR_RGB2BGR)) for v in views], 0).tolist() if views else None)
        return out

    def objects(self, x, y, yaw, hit=0.7, wrong=0.1, false=0.05):
        out, rng = [], self.rng
        for lm in self.world.landmarks:
            r, rad = math.hypot(lm["center"][0] - x, lm["center"][1] - y), max(lm["size"][:2]) / 2
            b = (math.atan2(lm["center"][1] - y, lm["center"][0] - x) - yaw + math.pi) % (2 * math.pi) - math.pi
            if r > 4.5 or abs(b) > self.hfov / 2 or rng.random() > hit:
                continue
            if self.world.raycast(np.array([x]), np.array([y]), np.array([yaw + b]))[0] < min(r - rad - 0.15, 4):
                continue  # hidden behind something
            out.append({"cls": lm["cls"] if rng.random() > wrong else str(rng.choice(self.classes)),
                        "bearing": b + rng.normal(0, math.radians(3)),
                        "range": max(r - 0.7 * rad, 0.2) * (1 + rng.normal(0, 0.1)) if rng.random() < 0.6 else None})
        if self.classes and rng.random() < false:
            out.append({"cls": str(rng.choice(self.classes)), "bearing": float(rng.uniform(-self.hfov / 2, self.hfov / 2)), "range": None})
        return out

    def free(self, x, y, clear=RADIUS):
        i, j = int((y - self.world.o[1]) / CELL) + 1, int((x - self.world.o[0]) / CELL) + 1
        return 0 <= i < self.blocked.shape[0] and 0 <= j < self.blocked.shape[1] and self.clearance[i, j] > clear

    def move(self, pose, v, w, dt):
        """Walk at v m/s, turn at w rad/s for dt s → (new true pose, odometry (forward, left, turn), bumped).
        Odometry: 4 % long, 2 % sideways, ~3°/m heading bias plus noise. Blocked by a wall or a leg, the duck feels the
        stall (its body stops, the IMU says so) and counts no forward motion, only the turn."""
        x, y, yaw = pose
        fwd, turn = v * dt, w * dt
        nyaw = yaw + turn
        nx, ny = x + fwd * math.cos(nyaw), y + fwd * math.sin(nyaw)
        bumped = not self.free(nx, ny)
        if bumped:
            nx, ny, fwd = x, y, 0.0
        rng = self.rng
        odo = (fwd * 1.04 + rng.normal(0, 0.02 * abs(fwd)), 0.02 * abs(fwd) + rng.normal(0, 0.03 * abs(fwd)),
               turn * 1.03 + 0.05 * abs(fwd) + rng.normal(0, 0.01 * abs(turn) + 0.01 * abs(fwd)))
        return (nx, ny, nyaw), odo, bumped

    def random_pose(self):
        """A free spot inside the rooms, facing anywhere: where someone might set the duck down."""
        cells = np.argwhere(~self.blocked & (self.world.room_of >= 0))
        i, j = cells[self.rng.integers(len(cells))]
        return ((j - 1 + 0.5) * CELL + self.world.o[0], (i - 1 + 0.5) * CELL + self.world.o[1], float(self.rng.uniform(-math.pi, math.pi)))
