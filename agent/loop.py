"""The duck goes somewhere on its own: odometry + ToF + markers → particle filter → A* route → steering.

Works against Pollen's duck-sim (or a robot) through agent/duck_rpc.py. From house-twin-patrol/:
uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python -m agent.loop --goal 2.0,0.9,-1.5708
"""
import argparse
import heapq
import json
import math
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from agent import perceive
from agent.duck_rpc import Duck
from agent.localize import CELL, HouseMap, ParticleFilter, _wrap

ROOT = Path(__file__).parent.parent
RADIUS, HZ, VX = 0.12, 5, 0.25  # the sim walking policy barely steps below 0.25 m/s


def route(hmap, start, goal):
    """A* on the map grown by the duck's radius, then pulled tight to straight runs between corners."""
    k = 2 * int(RADIUS / CELL) + 1
    blocked = cv.dilate(hmap.grid.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))).astype(bool)
    cell = lambda p: (int(p[1] / CELL) + 1, int(p[0] / CELL) + 1)
    s, g = cell(start), cell(goal)
    blocked[s] = blocked[g] = False
    came, cost, todo = {s: None}, {s: 0.0}, [(0.0, s)]
    while todo:
        _, c = heapq.heappop(todo)
        if c == g:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (c[0] + di, c[1] + dj)
                if (di or dj) and 0 <= n[0] < blocked.shape[0] and 0 <= n[1] < blocked.shape[1] and not blocked[n]:
                    new = cost[c] + math.hypot(di, dj)
                    if new < cost.get(n, 1e9):
                        cost[n], came[n] = new, c
                        heapq.heappush(todo, (new + math.hypot(g[0] - n[0], g[1] - n[1]), n))
    if g not in came:
        return None
    cells, c = [], g
    while c:
        cells.append(c)
        c = came[c]
    pts = [((j - 1 + 0.5) * CELL, (i - 1 + 0.5) * CELL) for i, j in cells[::-1]]
    clear = lambda a, b: all(not blocked[cell((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))] for t in np.linspace(0, 1, 40))
    out = [pts[0]]
    for i in range(1, len(pts)):
        if not clear(out[-1], pts[i]):
            out.append(pts[i - 1])
    return out[1:] + [pts[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True, help="x,y,heading in house metres/radians")
    ap.add_argument("--seconds", type=float, default=150)
    a = ap.parse_args()
    gx, gy, gyaw = map(float, a.goal.split(","))
    house = json.loads((ROOT / "out/current/house.json").read_text())
    spawn = json.loads((ROOT / "out/current/sim_frame.json").read_text())["spawn"]
    hmap = HouseMap(house)
    sizes = {m["id"]: m["size_m"] for m in house["markers"]}
    out = ROOT / "out/live"
    out.mkdir(parents=True, exist_ok=True)
    duck = Duck()
    while duck.state is None or duck.tof is None:
        time.sleep(0.1)
    pf = ParticleFilter(hmap, *spawn)
    last = duck.state["odom"]
    path = route(hmap, spawn[:2], (gx, gy))
    print("route:", [(round(x, 2), round(y, 2)) for x, y in path])
    trace, shots, seq, t0, done = open(out / "trace.jsonl", "w"), 0, None, time.time(), False
    while time.time() - t0 < a.seconds and not done:
        tick = time.time()
        od = duck.state["odom"]  # odometry deltas, turned into the duck's own frame
        dx, dy = od["position"][0] - last["position"][0], od["position"][1] - last["position"][1]
        c, s = math.cos(last["yaw"]), math.sin(last["yaw"])
        pf.predict(dx * c + dy * s, -dx * s + dy * c, float(_wrap(od["yaw"] - last["yaw"])))
        last = od
        if duck.tof["seq"] != seq:
            seq = duck.tof["seq"]
            pf.update_tof(duck.tof["distance_mm"], duck.tof["status"])
        found = []
        try:
            img, h = duck.frame()
            f = (img.shape[1] / 2) / math.tan(math.radians(45) / 2)  # head camera: 45° across the upright frame
            K = np.array([[f, 0, img.shape[1] / 2], [0, f, img.shape[0] / 2], [0, 0, 1]])
            found = perceive.markers(cv.cvtColor(img, cv.COLOR_BGR2GRAY), K, sizes)
            for m in found:
                pf.update_marker(m["id"], m["solutions"])
                if shots < 8:
                    cv.aruco.drawDetectedMarkers(img, [m["corners"][None]], np.array([[m["id"]]]))
                    cv.imwrite(str(out / f"duck_cam_{shots}.jpg"), img)
                    shots += 1
        except OSError:
            pass  # no camera running: odometry + ToF only
        x, y, yaw = pf.estimate()
        while path and math.hypot(path[0][0] - x, path[0][1] - y) < 0.15:
            path.pop(0)
        if path:
            err = float(_wrap(math.atan2(path[0][1] - y, path[0][0] - x) - yaw))
        else:
            err = float(_wrap(gyaw - yaw))
            done = abs(err) < math.radians(10)
        if abs(err) > math.radians(25) or not path:
            vx, vyaw = 0.0, float(np.clip(1.5 * err, -0.8, 0.8))
        else:
            vx, vyaw = VX, float(np.clip(1.2 * err, -0.6, 0.6))
        duck.move(0.0, 0.0) if done else duck.move(vx, vyaw)
        trace.write(json.dumps({"t": round(tick - t0, 2), "est": [round(x, 3), round(y, 3), round(yaw, 3)],
                                "odom": [round(v, 3) for v in od["position"][:2]] + [round(od["yaw"], 3)],
                                "markers": [m["id"] for m in found], "cmd": [vx, round(vyaw, 3)], "left": len(path)}) + "\n")
        time.sleep(max(0.0, 1 / HZ - (time.time() - tick)))
    for _ in range(5):
        duck.move(0.0, 0.0)
        time.sleep(0.1)
    print("arrived" if done else "timed out", "at", [round(v, 2) for v in pf.estimate()], f"after {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
