"""Drift test in the twin: a duck-height camera walks a known route; odometry with realistic error; OpenCV markers and the
simulated 8x8 ToF correct it. Compares odometry alone / +ToF / +markers / +both against the exact route.

uv run --python 3.12 --with mujoco --with opencv-python-headless==5.0.0.93 --with matplotlib python eval/localize_test.py [--imperfect-map]
--imperfect-map: the filter's map has every piece of furniture shifted up to 15 cm and misses a bag that is in the world
  (and, with --zero-stickers, every often-moved landmark such as a chair sits 1 m from where the map has it).
--walk <scan.zip>: the duck follows the path the person walked in that scan (out/current must be that scan's house);
  works for any home. No bag in the imperfect map (its spot belongs to the default route).
--zero-stickers: no markers; the duck starts with no idea where it is. Compares ToF alone / ToF + recognised objects
  (simulated detector: 70 % hit rate, 3° bearing noise, 10 % wrong class, 5 % false sightings) / the same from a known start.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2 as cv
import mujoco
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agent import perceive  # noqa: E402
from agent.localize import HouseMap, ParticleFilter  # noqa: E402

ROBOT = Path.home() / "Pollen/microduck_rl/src/mjlab_microduck/robot/microduck"
spec = importlib.util.spec_from_file_location("tof", ROBOT.parents[1] / "sim/tof.py")
tofmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tofmod)

house = json.loads((ROOT / "out/current/house.json").read_text())
frame = json.loads((ROOT / "out/current/sim_frame.json").read_text())
ZERO = "--zero-stickers" in sys.argv
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
OUT = ROOT / ("out/localize" + ("_zero" if ZERO else "") + ("_imperfect" if "--imperfect-map" in sys.argv else "")
              + (f"/seed{sys.argv[sys.argv.index('--seed') + 1]}" if "--seed" in sys.argv else ""))
OUT.mkdir(parents=True, exist_ok=True)
HZ, SPEED, TURN = 5, 0.25, 0.8
# hallway → kitchen walkway (look at #0) → back → hallway west (look at #2) → right corridor → living (look at #3)
ROUTE = [(3.3, 2.4), (3.45, 0.9), (2.0, 0.9), ("face", -math.pi / 2), (3.45, 0.95), (3.45, 2.3), (1.0, 2.3),
         ("face", math.pi), (3.45, 2.3), (3.45, 6.0), ("face", math.pi / 2), (1.0, 6.0)]
WALK = sys.argv[sys.argv.index("--walk") + 1] if "--walk" in sys.argv else None
if WALK:  # every 0.5 m of the person's walk that is free floor for a duck
    from twin.scan_io import Export
    sf = json.loads((ROOT / "out/current/report.json").read_text())["scan_frame"]
    c_, s_ = math.cos(sf["theta"]), math.sin(sf["theta"])
    free = HouseMap(house).free
    ROUTE = []
    for f in Export(WALK).frames:
        if f["tracking"] != "normal":
            continue
        T = np.array(f["T"]).reshape(4, 4)
        x, y = T[0, 3], -T[2, 3]  # ARKit → house axes, then the scan's turn and origin
        x, y = c_ * x + s_ * y - sf["origin"][0], -s_ * x + c_ * y - sf["origin"][1]
        if free(x, y) and (not ROUTE or math.hypot(x - ROUTE[-1][0], y - ROUTE[-1][1]) >= 0.5):
            ROUTE.append((float(x), float(y)))
    from agent.loop import route  # people squeeze past furniture: the duck plans its own way between their steps
    hm, legs = HouseMap(house), [ROUTE[0]]
    for b in ROUTE[3::3]:  # from where the duck got to, to the next step of theirs it can reach
        leg = route(hm, legs[-1], b)
        if leg:
            legs += leg[1:]
        elif len(legs) == 1:  # boxed in where the walk began: start from a later step instead
            legs = [b]
    ROUTE = legs

(ROBOT / "twin_eye.xml").write_text('''<mujoco model="twin_eye">
  <include file="myhouse.xml"/>
  <visual><global offwidth="800" offheight="800"/><headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/></visual>
  <worldbody>
    <light pos="0 0 2.9" dir="0 0 -1" directional="true" castshadow="false"/>
    <body name="eye" mocap="true" pos="0 0 0.2">
      <camera name="eye" fovy="57.7" xyaxes="0 -1 0 0 0 1"/>
      <site name="tof" size="0.01"/>
    </body>
    BAG
  </worldbody>
</mujoco>''')
sizes = {m["id"]: m["size_m"] for m in house["markers"]}
th, (px, py) = frame["theta"], frame["pos"]


def truth_path():
    x, y = ROUTE[0]
    yaw = math.atan2(ROUTE[1][1] - y, ROUTE[1][0] - x)
    poses = [(x, y, yaw)]
    for step in ROUTE[1:]:
        if step[0] == "face":
            goal = step[1]
        else:
            goal = math.atan2(step[1] - y, step[0] - x)
        while abs(d := (goal - yaw + math.pi) % (2 * math.pi) - math.pi) > 1e-3:  # turn on the spot
            yaw += max(-TURN / HZ, min(TURN / HZ, d))
            poses.append((x, y, yaw))
        if step[0] != "face":
            n = max(1, round(math.hypot(step[0] - x, step[1] - y) / (SPEED / HZ)))
            for k in range(1, n + 1):
                poses.append((x + (step[0] - x) * k / n, y + (step[1] - y) * k / n, yaw))
            x, y = step
    return poses


def sense(x, y, yaw):
    wx, wy = math.cos(th) * x - math.sin(th) * y + px, math.sin(th) * x + math.cos(th) * y + py
    data.mocap_pos[0] = [wx, wy, 0.2]
    data.mocap_quat[0] = [math.cos((yaw + th) / 2), 0, 0, math.sin((yaw + th) / 2)]
    mujoco.mj_forward(model, data)
    if ZERO:
        return None, [], tof.frame(data)
    renderer.update_scene(data, camera="eye")
    rgb = renderer.render()
    return rgb, perceive.markers(cv.cvtColor(rgb, cv.COLOR_RGB2GRAY), K, sizes), tof.frame(data)


rng = np.random.default_rng(7 + SEED)
truth = truth_path()
IMPERFECT = "--imperfect-map" in sys.argv
th, (px, py) = frame["theta"], frame["pos"]
if IMPERFECT:  # the filter's map is wrong; the world it senses is the real twin plus a bag nobody mapped
    mrng = np.random.default_rng(11)
    fmap = json.loads(json.dumps(house))
    for o in fmap["objects"] + fmap.get("landmarks", []):
        o["center"] = [c + float(mrng.uniform(-0.15, 0.15)) for c in o["center"]]
    hmap = HouseMap(fmap)
    bag = (3.2, 4.4) if not WALK else (-50.0, -50.0)  # beside the right corridor the route uses, 15 cm clear of the duck's line (it can't drive through it)
    bx_, by_ = math.cos(th) * bag[0] - math.sin(th) * bag[1] + px, math.sin(th) * bag[0] + math.cos(th) * bag[1] + py
else:
    hmap = HouseMap(house)
world = HouseMap(house)  # what the duck's camera really sees
if IMPERFECT and ZERO:  # chairs and other often-moved things stand 1 m from where the map has them
    world.landmarks = [dict(lm) for lm in world.landmarks]
    for lm in world.landmarks:
        if lm["mobility"] == "often":
            a = float(mrng.uniform(-math.pi, math.pi))
            lm["center"] = [lm["center"][0] + math.cos(a), lm["center"][1] + math.sin(a)]
HFOV = 2 * math.atan(240 / (320 / math.tan(math.radians(57.7) / 2)))
CLASSES = sorted({lm["cls"] for lm in world.landmarks})


def seen_objects(x, y, yaw):
    """A stand-in for YOLOX on the duck camera: the true landmarks in view and not hidden, with realistic mistakes."""
    out = []
    for lm in world.landmarks:
        r, rad = math.hypot(lm["center"][0] - x, lm["center"][1] - y), max(lm["size"][:2]) / 2
        b = (math.atan2(lm["center"][1] - y, lm["center"][0] - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        if r > 4.5 or abs(b) > HFOV / 2 or rng.random() > 0.7 or world.raycast(np.array([x]), np.array([y]), np.array([yaw + b]))[0] < min(r - rad - 0.15, 4):
            continue
        out.append({"cls": lm["cls"] if rng.random() > 0.1 else str(rng.choice(CLASSES)), "bearing": b + rng.normal(0, math.radians(3)),
                    "range": max(r - 0.7 * rad, 0.2) * (1 + rng.normal(0, 0.1)) if rng.random() < 0.6 else None})
    if rng.random() < 0.05:
        out.append({"cls": str(rng.choice(CLASSES)), "bearing": float(rng.uniform(-HFOV / 2, HFOV / 2)), "range": None})
    return out

xml = (ROBOT / "twin_eye.xml").read_text().replace("BAG", f'<geom type="box" pos="{bx_:.3f} {by_:.3f} 0.15" size="0.1 0.15 0.15" rgba="0.2 0.3 0.6 1"/>' if IMPERFECT else "")
(ROBOT / f"twin_eye_run{'_imperfect' if IMPERFECT else ''}{SEED}.xml").write_text(xml)
model = mujoco.MjModel.from_xml_path(str(ROBOT / f"twin_eye_run{'_imperfect' if IMPERFECT else ''}{SEED}.xml"))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 640, 480)
tof = tofmod.Tof(model, model.site("tof").id, seed=1)
fy = 320 / math.tan(math.radians(model.cam_fovy[0]) / 2)
K = np.array([[fy, 0, 240], [0, fy, 320], [0, 0, 1]])
x0, y0, yaw0 = truth[0]
if ZERO:
    filters = {"ToF, unknown start": ParticleFilter.anywhere(hmap, seed=3 + SEED), "ToF + objects, unknown start": ParticleFilter.anywhere(hmap, seed=3 + SEED),
               "ToF + objects, known start": ParticleFilter(hmap, x0, y0, yaw0, seed=3 + SEED)}
else:
    filters = {name: ParticleFilter(hmap, x0, y0, yaw0, seed=3) for name in ("+ToF", "+markers", "+both")}
TRACK = "ToF + objects, unknown start" if ZERO else "+both"
dead = [x0, y0, yaw0]
log = {name: [] for name in ("odometry only", *filters)}
seen, shots = [], 0
for k in range(1, len(truth)):
    (ax, ay, ayaw), (bx, by, byaw) = truth[k - 1], truth[k]
    fwd = (bx - ax) * math.cos(ayaw) + (by - ay) * math.sin(ayaw)
    turn = (byaw - ayaw + math.pi) % (2 * math.pi) - math.pi
    # the duck's own odometry: 4 % long, drifts 2 % sideways, ~3°/m heading bias, plus noise
    of = fwd * 1.04 + rng.normal(0, 0.02 * abs(fwd))
    ol = 0.02 * abs(fwd) + rng.normal(0, 0.03 * abs(fwd))
    ot = turn * 1.03 + 0.05 * abs(fwd) + rng.normal(0, 0.01 * abs(turn) + 0.01 * abs(fwd))
    dead[0] += of * math.cos(dead[2]) - ol * math.sin(dead[2])
    dead[1] += of * math.sin(dead[2]) + ol * math.cos(dead[2])
    dead[2] += ot
    rgb, found, (dist, status) = sense(bx, by, byaw)
    objs = seen_objects(bx, by, byaw) if ZERO else []
    for name, f in filters.items():
        f.predict(of, ol, ot)
        if "ToF" in name or name == "+both":
            f.update_tof(dist, status)
        if "objects" in name:
            f.update_landmarks(objs, HFOV)
        if name in ("+markers", "+both"):
            for m in found:
                f.update_marker(m["id"], m["solutions"])
    for m in found:
        seen.append((k, m["id"], float(np.linalg.norm(m["solutions"][0][1]))))
        if shots < 6 and k % 5 == 0:  # a few duck-eye frames with the detection drawn, for the report
            img = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            cv.aruco.drawDetectedMarkers(img, [m["corners"][None]], np.array([[m["id"]]]))
            R, t, _ = m["solutions"][0]
            cv.drawFrameAxes(img, K, None, cv.Rodrigues(R)[0], t, 0.08)
            cv.putText(img, f'#{m["id"]}  {np.linalg.norm(t):.2f} m', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv.imwrite(str(OUT / f"duck_eye_{shots}.jpg"), img)
            shots += 1
    for name, est in [("odometry only", dead), *((n, f.estimate()) for n, f in filters.items())]:
        log[name].append(math.hypot(est[0] - bx, est[1] - by))
        if name == TRACK:
            log.setdefault("_track", []).append((bx, by, est[0], est[1], dead[0], dead[1]))

t = np.arange(1, len(truth)) / HZ
summary = {name: {"median_m": round(float(np.median(e)), 3), "p90_m": round(float(np.percentile(e, 90)), 3),
                  "max_m": round(float(np.max(e)), 3), "final_m": round(float(e[-1]), 3)}
           for name, e in log.items() if not name.startswith("_")}
for name, e in log.items():  # lock = first moment the guess stays within 0.3 m for 2 s; then how well it holds
    if name.startswith("_"):
        continue
    ok = np.convolve(np.array(e) < 0.3, np.ones(2 * HZ), "valid") == 2 * HZ
    lock = int(np.argmax(ok)) if ok.any() else None
    summary[name].update({"lock_s": round(lock / HZ, 1) if lock is not None else None,
                          "after_lock_median_m": round(float(np.median(e[lock:])), 3) if lock is not None else None,
                          "after_lock_max_m": round(float(np.max(e[lock:])), 3) if lock is not None else None,
                          "within_0.3m_pct": round(100 * float(np.mean(np.array(e) < 0.3)), 1)})
summary["route"] = {"seconds": round(float(t[-1]), 1), "metres": round(float(sum(math.hypot(truth[i][0] - truth[i - 1][0], truth[i][1] - truth[i - 1][1]) for i in range(1, len(truth)))), 1),
                    "frames": len(truth) - 1, "marker_sightings": len(seen), "marker_ids": sorted({s[1] for s in seen}),
                    "sighting_range_m": [round(min(s[2] for s in seen), 2), round(max(s[2] for s in seen), 2)] if seen else None}
(OUT / "results.json").write_text(json.dumps(summary, indent=1))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
for name, e in log.items():
    if not name.startswith("_"):
        a1.plot(t, e, label=f'{name} (median {summary[name]["median_m"]:.2f} m)')
for k, i, _ in seen:
    a1.axvline(k / HZ, color="green", alpha=0.08)
a1.set_xlabel("time (s)"); a1.set_ylabel("position error (m)"); a1.set_ylim(0, 4)
a1.set_title("Duck position error, no stickers, unknown start" if ZERO else "Duck position error on the route (green = marker in view)"); a1.legend()
tr = np.array(log["_track"])
a2.imshow(~hmap.grid[::-1], cmap="gray", extent=(-0.02, hmap.W + 0.04, -0.02, hmap.L + 0.04), alpha=0.5)
a2.plot(tr[:, 0], tr[:, 1], "k-", lw=2, label="true path")
a2.plot(tr[:, 4], tr[:, 5], "r--", label="odometry only")
a2.plot(tr[:, 2], tr[:, 3], "b-", label=f"filter ({TRACK})")
for lm in world.landmarks:
    a2.plot(*lm["center"], "*", color="orange", ms=6 + 8 * lm["weight"])
for m in house["markers"]:
    a2.plot(m["x"], m["y"], "gs"); a2.annotate(f'#{m["id"]}', (m["x"], m["y"]), color="green")
a2.set_aspect("equal"); a2.set_title("Seen from above (m)"); a2.legend(loc="center right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "drift.png", dpi=110)
print(json.dumps(summary, indent=1))
