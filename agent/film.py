"""Film of a live walk (agent/walk.py): the recorded body (agent/duck_recorder.py) replayed in the same MuJoCo scene.
Chase camera + the duck's own eye + the map (truth vs where the duck thinks it is) + what the brain is doing.

python -m agent.film --house out/houses/<name> [--speed 1]  →  out/walk/<house>/walk.mp4
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2 as cv
import mujoco
import numpy as np

from agent.twin_sim import ROBOT, ROOT

WIRE = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw",
        "head_roll", "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
MODE = {"look": ((80, 190, 255), "looking around"), "go": ((110, 220, 120), "walking to goal"),
        "back off": ((60, 140, 255), "backing off"), "stuck": ((80, 80, 240), "no route")}
FONT = cv.FONT_HERSHEY_SIMPLEX


def text(img, s, xy, scale=0.6, color=(235, 235, 235), thick=1):
    cv.putText(img, s, xy, FONT, scale, color, thick, cv.LINE_AA)


def map_panel(house, goals, trace, k, size=(640, 600)):
    xs = [p for w in house["walls"] for p in (w["a"][0], w["b"][0])]
    ys = [p for w in house["walls"] for p in (w["a"][1], w["b"][1])]
    lo, hi = np.array([min(xs), min(ys)]) - 0.2, np.array([max(xs), max(ys)]) + 0.2
    sc = min((size[0] - 20) / (hi[0] - lo[0]), (size[1] - 60) / (hi[1] - lo[1]))
    px = lambda p: (int(10 + (p[0] - lo[0]) * sc), int(size[1] - 10 - (p[1] - lo[1]) * sc))  # noqa: E731
    img = np.full((size[1], size[0], 3), 28, np.uint8)
    for r, col in zip(house["rooms"], [(52, 46, 40), (40, 50, 44), (46, 40, 52), (40, 46, 52)]):
        for x0, y0, x1, y1 in r["rects"]:
            cv.rectangle(img, px((x0, y0)), px((x1, y1)), col, -1)
    for o in house["objects"]:
        w, d = o["size"][0] / 2, o["size"][1] / 2
        c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
        pts = np.array([px((o["center"][0] + a * c - b * s, o["center"][1] + a * s + b * c)) for a, b in ((-w, -d), (w, -d), (w, d), (-w, d))])
        cv.fillPoly(img, [pts], (78, 78, 84))
    for w in house["walls"]:
        cv.line(img, px(w["a"]), px(w["b"]), (210, 210, 205), 3)
    for r in house["rooms"]:
        x0, y0, x1, y1 = max(r["rects"], key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        text(img, r["name"].replace("_", " "), (px((x0, y1))[0] + 6, px((x0, y1))[1] + 20), 0.5, (170, 170, 165))
    steps = [s for s in trace[:k + 1] if "mode" in s]
    if len(steps) > 1:
        cv.polylines(img, [np.array([px(s["true"]) for s in steps])], False, (90, 200, 255), 2, cv.LINE_AA)
    cur = next((trace[j] for j in range(k, -1, -1) if "mode" in trace[j]), trace[k])  # a step, not an event line
    for g in goals:
        done = any(s.get("event") == "arrive" and s["goal"] == g["name"] for s in trace[:k + 1])
        col = (110, 220, 120) if done else (255, 255, 255) if g["name"] == cur.get("goal") else (130, 130, 130)
        cv.drawMarker(img, px(g["xy"]), col, cv.MARKER_TILTED_CROSS if not done else cv.MARKER_STAR, 22, 2)
    if "est" in cur:
        t, e = cur["true"], cur["est"]
        cv.circle(img, px(t), 9, (90, 200, 255), -1, cv.LINE_AA)
        cv.line(img, px(t), px((t[0] + 0.3 * math.cos(t[2]), t[1] + 0.3 * math.sin(t[2]))), (90, 200, 255), 3, cv.LINE_AA)
        a = int(255 * min(1, cur["conf"]))
        cv.circle(img, px(e), 13, (255, 120 + a // 2, 255 - a), 2, cv.LINE_AA)
        cv.line(img, px(e), px((e[0] + 0.35 * math.cos(e[2]), e[1] + 0.35 * math.sin(e[2]))), (255, 120 + a // 2, 255 - a), 2, cv.LINE_AA)
        if cur.get("aim"):
            cv.line(img, px(e), px(cur["aim"]), (110, 220, 120), 1, cv.LINE_AA)
    text(img, "map from the LiDAR scan", (12, 24), 0.55)
    cv.circle(img, (12 + 6, 44), 7, (90, 200, 255), -1)
    text(img, "where it really is", (32, 49), 0.5, (90, 200, 255))
    cv.circle(img, (230, 44), 9, (255, 220, 120), 2)
    text(img, "where it thinks it is", (246, 49), 0.5, (255, 220, 120))
    return img


def main():
    ap = argparse.ArgumentParser(prog="agent.film")
    ap.add_argument("--house", default=str(ROOT / "out/current"))
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    house_dir = Path(a.house).resolve()
    out = ROOT / "out/walk" / house_dir.name
    body = np.load(out / "body.npy")
    trace = [json.loads(s) for s in (out / "trace.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    house = json.loads((house_dir / "house.json").read_text())
    walls = [s.get("wall") for s in trace]
    w0 = next(w for w in walls if w)
    for i, s in enumerate(trace):  # events carry only brain time: give them the wall time of the step before
        if not s.get("wall"):
            s["wall"] = trace[i - 1]["wall"] if i else w0
    wall = np.array([s["wall"] for s in trace])

    m = mujoco.MjModel.from_xml_path(str(ROBOT / "scene_myhouse.xml"))
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] if n != "mouth" else None for n in WIRE]
    chase = mujoco.Renderer(m, 720, 1280)
    eye = mujoco.Renderer(m, 360, 640)
    # the chase camera looks through walls and tall furniture, the duck's eye doesn't
    tall = np.where((m.geom_bodyid == m.body("house").id) & (m.geom_size[:, 2] > 0.05))[0]
    solid, wall_mat = m.geom_rgba[tall, 3].copy(), m.mat("wall").id
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 1.6, -48
    fps = 25
    times = np.arange(max(body[0, 0], w0), min(body[-1, 0], wall[-1] + 3), a.speed / fps)
    raw = out / "walk_raw.mp4"
    vw = cv.VideoWriter(str(raw), cv.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    az = None
    for n, tw in enumerate(times):
        row = body[min(np.searchsorted(body[:, 0], tw), len(body) - 1)]
        d.qpos[:7] = row[1:8]
        for q, v in zip(adr, row[8:]):
            if q is not None:
                d.qpos[q] = v
        mujoco.mj_forward(m, d)
        w, x, y, z = row[4:8]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        az = yaw if az is None else az + 0.04 * ((yaw - az + 180) % 360 - 180)  # a camera operator who follows, gently
        cam.lookat[:] = [row[1], row[2], 0.12]
        cam.azimuth = az + 180 - 25
        m.geom_rgba[tall, 3], m.mat_rgba[wall_mat, 3] = 0.3, 0.2
        chase.update_scene(d, cam)
        m.geom_rgba[tall, 3], m.mat_rgba[wall_mat, 3] = solid, 1.0
        frame = np.zeros((1080, 1920, 3), np.uint8)
        frame[:720, :1280] = cv.cvtColor(chase.render(), cv.COLOR_RGB2BGR)
        eye.update_scene(d, "head_camera")
        frame[:360, 1280:1920] = cv.cvtColor(eye.render(), cv.COLOR_RGB2BGR)
        k = max(0, int(np.searchsorted(wall, tw)) - 1)
        s = next(trace[j] for j in range(k, -1, -1) if "mode" in trace[j]) if any("mode" in t for t in trace[:k + 1]) else trace[0]
        frame[480:1080, 1280:1920] = map_panel(house, summary["goal_points"], trace, k)
        frame[360:480, 1280:1920] = 20
        text(frame, "^ the duck's own eye (head camera, 20 cm above the floor)", (1296, 400), 0.55)
        # the brain, in words
        strip = frame[720:1080, :1280]
        strip[:] = 20
        col, what = MODE.get(s.get("mode"), ((200, 200, 200), "standing up"))
        t_brain = tw - w0
        text(strip, f"{t_brain:5.1f} s", (24, 46), 1.0)
        text(strip, what, (190, 46), 1.0, col, 2)
        text(strip, f"next goal: {s.get('goal', '').replace('_', ' ')}", (720, 46), 0.8)
        conf = s.get("conf", 0)
        cv.rectangle(strip, (24, 76), (424, 100), (60, 60, 60), -1)
        cv.rectangle(strip, (24, 76), (24 + int(400 * conf), 100), (110, 220, 120) if conf >= 0.6 else (80, 190, 255), -1)
        text(strip, f"sure of its position: {conf:.0%}", (440, 96), 0.7)
        said, true_room = s.get("room_said", "").replace("_", " "), s.get("room_true", "").replace("_", " ")
        if conf >= 0.6:
            text(strip, f"it says: I'm in the {said}", (24, 140), 0.8, (110, 220, 120) if said == true_room else (80, 80, 240), 2)
        else:
            text(strip, "it says: not sure where I am yet", (24, 140), 0.8, (80, 190, 255), 2)
        text(strip, f"(truth: {true_room})", (640, 140), 0.7, (170, 170, 165))
        seen = sorted(set(s.get("seen", []))) + ([f"colours like the {s['colour'].replace('_', ' ')}"] if s.get("colour") else [])
        if seen:
            text(strip, "sees: " + ", ".join(seen), (24, 180), 0.7, (220, 200, 150))
        if s.get("cmd"):
            text(strip, f"command to the walking policy: forward {s['cmd'][0]:.2f} m/s, turn {s['cmd'][1]:+.2f} rad/s", (24, 220), 0.6, (170, 170, 165))
        done = [t for t in trace[:k + 1] if t.get("event") == "arrive"]
        text(strip, "reached: " + ("  ".join(f"{t['goal'].replace('_', ' ')} ({t['true_dist'] * 100:.0f} cm off)" for t in done) or "-"),
             (24, 262), 0.7, (110, 220, 120))
        ev = next((t for t in trace[k:max(0, k - 60):-1] if t.get("event") in ("vision_decision", "approval")), None)
        if ev and tw - ev["wall"] < 4:  # OpenCV changed what the brain does, or it asked a person
            msg = (f"OpenCV changed the plan [{ev['cause']}]: {ev['before']['mode']} -> {ev['after']['mode']}, "
                   f"thinks it's in: {ev['after']['room'] or 'a doorway'}".replace("_", " ") if ev["event"] == "vision_decision"
                   else f"asked a person: keep looking? -> {'yes' if ev['answer'] else 'no, stop'} ({ev['by']})")
            text(strip, msg, (24, 292), 0.65, (120, 200, 255), 2)
        text(strip, "Pollen Microduck - real robotd + trained walking policy (ONNX, 50 Hz) in MuJoCo - zero stickers", (24, 316), 0.6, (150, 150, 150))
        text(strip, f"house: built from an iPhone LiDAR scan - {a.speed:g}x speed", (24, 344), 0.6, (150, 150, 150))
        vw.write(frame)
        if n % 250 == 0:
            print(f"{n}/{len(times)}", flush=True)
    vw.release()
    final = out / ("walk.mp4" if a.speed == 1 else f"walk_{a.speed:g}x.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                    "-movflags", "+faststart", str(final)], check=True)
    raw.unlink()
    print(final)


if __name__ == "__main__":
    main()
