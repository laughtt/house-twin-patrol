"""house.json → MuJoCo house + scene with Pollen's Microduck, in the style of microduck_rl's apartment.xml.

python -m twin sim [house.json] --spawn x,y,heading   (default: out/current/house.json)
Writes myhouse.xml + scene_myhouse.xml next to the robot XML (so its includes and meshes resolve), sim_frame.json and two
previews next to the house.json, and points out/current at that house.
"""
import argparse
import json
import math
import re
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ROBOT = Path(CFG["paths"]["pollen_rl"]).expanduser() / "src/mjlab_microduck/robot/microduck"
# one structurally different floor per zone (Pollen: distinct floors stop rooms looking alike to a low camera)
FLOORS = [('mark="cross"', "0.76 0.60 0.36", "0.48 0.36 0.20", 3), ('mark="random" random="0.55"', "0.90 0.92 0.86", "0.58 0.66 0.56", 6),
          ('mark="edge"', "0.55 0.52 0.50", "0.34 0.32 0.31", 9), ('mark="random" random="0.6"', "0.72 0.64 0.82", "0.50 0.42 0.62", 2)]
COLOR = {"counter": "0.55 0.56 0.56 1", "peninsula": "0.55 0.56 0.56 1", "fridge": "0.93 0.93 0.92 1", "sofa": "0.86 0.82 0.74 1",
         "table": "0.55 0.36 0.20 1", "chair": "0.62 0.44 0.28 1"}


def color(label):
    return next((c for k, c in COLOR.items() if k in label), "0.7 0.7 0.7 1")


def legs(cx, cy, w, d, top, yaw, rgba, back=0.0):
    """Top slab + 4 legs (+ a backrest for chairs): the duck can walk under it, like the real thing."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = lambda x, y: (cx + x * c - y * s, cy + x * s + y * c)
    out = [f'<geom type="box" pos="{cx:.3f} {cy:.3f} {top - 0.02:.3f}" size="{w / 2:.3f} {d / 2:.3f} 0.02" euler="0 0 {yaw:.4f}" rgba="{rgba}"/>']
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = rot(sx * (w / 2 - 0.04), sy * (d / 2 - 0.04))
            out.append(f'<geom type="box" pos="{x:.3f} {y:.3f} {(top - 0.04) / 2:.3f}" size="0.02 0.02 {(top - 0.04) / 2:.3f}" rgba="{rgba}"/>')
    if back:
        x, y = rot(0, d / 2 - 0.02)
        out.append(f'<geom type="box" pos="{x:.3f} {y:.3f} {top + back / 2:.3f}" size="{w / 2:.3f} 0.02 {back / 2:.3f}" euler="0 0 {yaw:.4f}" rgba="{rgba}"/>')
    return out


def house_xml(h, frame):
    xs = [p for w in h["walls"] for p in (w["a"][0], w["b"][0])]
    ys = [p for w in h["walls"] for p in (w["a"][1], w["b"][1])]
    W, L = max(xs), max(ys)
    assets = ['<texture type="2d" name="ft_main" builtin="checker" mark="edge" rgb1="0.80 0.78 0.74" rgb2="0.70 0.68 0.64" markrgb="0.5 0.5 0.5" width="512" height="512"/>',
              '<material name="floor_main" texture="ft_main" texrepeat="8 6" reflectance="0.03"/>',
              '<texture type="2d" name="wt" builtin="checker" mark="random" random="0.3" rgb1="0.93 0.92 0.89" rgb2="0.85 0.84 0.80" markrgb="0.6 0.6 0.6" width="512" height="512"/>',
              '<material name="wall" texture="wt" texrepeat="4 1" reflectance="0.02"/>']
    geoms = [f'<geom name="floor" type="box" pos="{W / 2:.3f} {L / 2:.3f} -0.005" size="{W / 2:.3f} {L / 2:.3f} 0.005" material="floor_main"/>']
    top = max(w["h"] for w in h["walls"])  # ceiling plane faces down: the duck sees it, the bird's-eye view does not
    geoms.append(f'<geom name="ceiling" type="plane" pos="{W / 2:.3f} {L / 2:.3f} {top:.3f}" size="{W / 2:.3f} {L / 2:.3f} 0.1" euler="3.1416 0 0" rgba="0.95 0.95 0.93 1" contype="0" conaffinity="0"/>')
    for i, r in enumerate(h["rooms"]):
        mark, c1, c2, rep = FLOORS[i % len(FLOORS)]
        assets += [f'<texture type="2d" name="ft{i}" builtin="checker" {mark} rgb1="{c1}" rgb2="{c2}" markrgb="0.2 0.2 0.2" width="512" height="512"/>',
                   f'<material name="floor{i}" texture="ft{i}" texrepeat="{rep} {rep}" reflectance="0.04"/>']
        poly = np.array(r["polygon"])
        for k, (x0, y0, x1, y1) in enumerate(r.get("rects") or [list(poly.min(0)) + list(poly.max(0))]):  # a zone = its rectangles
            geoms.append(f'<geom name="zone_{re.sub(r"[^a-z0-9]+", "_", r["name"].lower())}_{k}" type="box" pos="{(x0 + x1) / 2:.3f} {(y0 + y1) / 2:.3f} 0.0005" '
                         f'size="{(x1 - x0) / 2:.3f} {(y1 - y0) / 2:.3f} 0.0005" material="floor{i}" contype="0" conaffinity="0"/>')
    for w in h["walls"]:  # the wall line is the measured surface: the wall's thickness goes outward
        (ax, ay), (bx, by) = w["a"], w["b"]
        n = math.hypot(bx - ax, by - ay)
        ox, oy = (ay - by) / n, (bx - ax) / n
        if ox * ((ax + bx) / 2 - W / 2) + oy * ((ay + by) / 2 - L / 2) < 0:
            ox, oy = -ox, -oy
        push = 0 if w.get("shared") else w["t"] / 2  # a wall between two rooms is already its centre line
        mx, my = (ax + bx) / 2 + ox * push, (ay + by) / 2 + oy * push
        geoms.append(f'<geom type="box" pos="{mx:.3f} {my:.3f} {w["h"] / 2:.3f}" size="{n / 2 + (0 if w.get("shared") else w["t"]):.3f} {w["t"] / 2:.3f} {w["h"] / 2:.3f}" '
                     f'euler="0 0 {math.atan2(by - ay, bx - ax):.4f}" material="wall"/>')
    for o in h["objects"]:
        (cx, cy), (w, d, top), yaw, rgba = o["center"], o["size"], o["yaw"], color(o["label"])
        if o["kind"] == "table" or o["label"] == "chair":
            geoms += legs(cx, cy, w, d, 0.46 if o["label"] == "chair" else top, yaw, rgba, back=top - 0.46 if o["label"] == "chair" else 0)
        else:
            geoms.append(f'<geom type="box" pos="{cx:.3f} {cy:.3f} {top / 2:.3f}" size="{w / 2:.3f} {d / 2:.3f} {top / 2:.3f}" euler="0 0 {yaw:.4f}" rgba="{rgba}"/>')
    for m in h.get("markers", []):  # ArUco plates (white border of one cell), 2 mm proud of the surface they're on
        import cv2 as cv
        if not (ROBOT / f'aruco_{m["id"]}.png').exists():  # rewriting it races twins built side by side
            img = cv.aruco.generateImageMarker(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), m["id"], 600)
            cv.imwrite(str(ROBOT / f'aruco_{m["id"]}.png'), cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_CONSTANT, value=255))
        assets += [f'<texture type="2d" name="aruco{m["id"]}" file="aruco_{m["id"]}.png"/>',
                   f'<material name="aruco{m["id"]}" texture="aruco{m["id"]}" texrepeat="1 1"/>']
        nx, ny, half = math.cos(m["yaw"]), math.sin(m["yaw"]), m["size_m"] * 8 / 12
        geoms.append(f'<geom name="marker_{m["id"]}" type="box" pos="{m["x"] + 0.002 * nx:.4f} {m["y"] + 0.002 * ny:.4f} {m["z"]:.3f}" '
                     f'size="{half:.4f} {half:.4f} 0.001" xyaxes="{-ny:.4f} {nx:.4f} 0 0 0 1" material="aruco{m["id"]}" contype="0" conaffinity="0"/>')
    body = "\n      ".join(geoms)
    return f'''<!-- Generated by house-twin-patrol/twin/to_mjcf.py from house.json: {h["source"]} -->
<mujoco model="myhouse">
  <compiler angle="radian"/>
  <asset>
    {chr(10).join("    " + a for a in assets).strip()}
  </asset>
  <worldbody>
    <body name="house" pos="{frame["pos"][0]:.4f} {frame["pos"][1]:.4f} 0" euler="0 0 {frame["theta"]:.5f}">
      {body}
    </body>
  </worldbody>
</mujoco>
''', W, L


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twin sim")
    ap.add_argument("house", nargs="?", default=str(ROOT / "out/current/house.json"))
    ap.add_argument("--spawn", default="", help="x,y,heading the duck faces in radians (default: centre of the first zone)")
    ap.add_argument("--no-markers", action="store_true", help="zero stickers: leave the scanned markers out")
    a = ap.parse_args(argv)
    house_path = Path(a.house).resolve()
    here = house_path.parent
    h = json.loads(house_path.read_text())
    if a.spawn:
        x, y, yaw = map(float, a.spawn.split(","))
    else:
        (x0, y0), (x1, y1) = h["rooms"][0]["polygon"][0], h["rooms"][0]["polygon"][2]
        x, y, yaw = (x0 + x1) / 2, (y0 + y1) / 2, 0.0
    # duck-body always places the duck at the world origin facing +x, so the house is turned and shifted instead:
    # world = R(theta) · (house - spawn). Saved so the agent can convert duck poses back to house coordinates.
    theta = -yaw
    c, s = math.cos(theta), math.sin(theta)
    frame = {"spawn": [x, y, yaw], "theta": theta, "pos": [-(c * x - s * y), -(s * x + c * y)]}
    (here / "sim_frame.json").write_text(json.dumps(frame))
    current = ROOT / "out/current"
    if current.is_symlink() or not current.exists():
        current.unlink(missing_ok=True)
        current.symlink_to(here)
    xml, W, L = house_xml({**h, "markers": []} if a.no_markers else h, frame)
    (ROBOT / "myhouse.xml").write_text(xml)
    scene = (ROBOT / "scene_apartment.xml").read_text().replace("apartment.xml", "myhouse.xml")
    scene = scene.replace('<global azimuth="160" elevation="-20" />', '<global azimuth="160" elevation="-20" offwidth="1280" offheight="720"/>')
    scene = scene.replace('directional="true"', 'directional="true" castshadow="false"')  # else the ceiling darkens the house
    (ROBOT / "scene_myhouse.xml").write_text(scene)
    print(ROBOT / "scene_myhouse.xml", f"house {W:.2f} x {L:.2f} m, spawn {x:.2f},{y:.2f}")

    import cv2 as cv
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(ROBOT / "scene_myhouse.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [c * W / 2 - s * L / 2 + frame["pos"][0], s * W / 2 + c * L / 2 + frame["pos"][1], 0]
    cam.distance, cam.elevation, cam.azimuth = max(W, L) * 1.35, -68, 90 + math.degrees(theta)
    r.update_scene(d, cam)
    cv.imwrite(str(here / "twin_overview.png"), r.render()[:, :, ::-1])
    eye = mujoco.MjvCamera()  # on the duck's head, looking along its walking direction (+x); 45° wide like the head camera
    eye.lookat[:] = [1.05, 0.0, 0.2]
    eye.distance, eye.elevation, eye.azimuth = 1.0, 0.0, 0.0
    r2 = mujoco.Renderer(m, 640, 360)
    r2.update_scene(d, eye)
    cv.imwrite(str(here / "twin_duck_eye.png"), r2.render()[:, :, ::-1])
    print(f"previews: {here}/twin_overview.png, twin_duck_eye.png; out/current → {here}")


if __name__ == "__main__":
    main()
