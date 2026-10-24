"""Zero-sticker localization on real images: the scan's own duck-height frames play the duck's camera.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 --with matplotlib python eval/lowview_test.py <scan.zip> [--all-frames]

- map: out/houses/<session>/house.json + report.json (`python -m twin build` of the same scan). Optimistic: same day,
  same light, and the landmarks were found in these very frames.
- duck camera: every frame lower than [landmarks] low_h (--all-frames: every frame, to try the harness on an old scan).
  YOLOX on the upright picture → class + bearing (the pixel's ray turned into the house frame); range from LiDAR depth.
- duck ToF: the LiDAR depth within ±5° of level, in 8 columns over 45° around the heading (no data → skipped).
- odometry: ARKit pose deltas with the drift of eval/localize_test.py (4 % long, 2 % sideways, ~3°/m heading).
- truth: the ARKit pose. Every filter starts with no idea where it is, except the "known start" one.
"""
import json
import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agent.localize import HouseMap, ParticleFilter, _wrap  # noqa: E402
from agent.perceive import colour_hist  # noqa: E402
from twin import CFG  # noqa: E402
from twin.landmarks import upright_detect  # noqa: E402
from twin.fuse import ground_at  # noqa: E402
from twin.scan_io import Export, kmat  # noqa: E402

ex = Export(sys.argv[1])
ALL = "--all-frames" in sys.argv
ODO = sys.argv[sys.argv.index("--odometry") + 1] if "--odometry" in sys.argv else "duck"  # duck | gyro | perfect
HOUSE = ROOT / "out/houses" / ex.session
house = json.loads((HOUSE / "house.json").read_text())
sf = json.loads((HOUSE / "report.json").read_text())["scan_frame"]
OUT = ROOT / "out/lowview" / (ex.session + ("-all" if ALL else "") + ("" if ODO == "duck" else "-" + ODO))
OUT.mkdir(parents=True, exist_ok=True)
c, s = math.cos(sf["theta"]), math.sin(sf["theta"])
R2, z0, o = np.array([[c, s], [-s, c]]), sf["z0"], np.array(sf["origin"])
frames = [f for f in ex.frames if f["tracking"] == "normal"]
_z = np.load(HOUSE / house["occupancy"]["file"]) if "occupancy" in house else {}
GROUND = (_z["ground_lo"], float(_z["ground_tile"]), _z["ground"]) if "ground" in _z else None


def pose(P):
    """ARKit-derived camera pose → house frame: position (x, y, height) and the heading on the floor."""
    p = P[:3, 3].copy()
    p[2] -= z0
    p[:2] = R2 @ p[:2] - o
    if GROUND is not None:  # height above the floor under the camera (the scan's floor tiles)
        p[2] -= float(ground_at(GROUND, p[:2]))
    fwd = R2 @ (P[:3, :3] @ [0, 0, 1.0])[:2]
    return p, math.atan2(fwd[1], fwd[0])


def to_house_dirs(P, rays):
    d = rays @ P[:3, :3].T
    d[:, :2] = d[:, :2] @ R2.T
    return d


def tof(P3, d, P, cam, yaw):
    """8 zones × 45° of level LiDAR returns, as the duck's 8x8 sensor rows 3-4 would see them (mm, status)."""
    q = P3[..., :3].reshape(-1, 3)[d.ravel() > 0]
    v = to_house_dirs(P, q)
    horiz = np.hypot(v[:, 0], v[:, 1])
    az = _wrap(np.arctan2(v[:, 1], v[:, 0]) - yaw)
    level = np.abs(np.arctan2(v[:, 2], horiz)) < np.radians(5)
    col = np.floor((np.radians(22.5) - az) / np.radians(45 / 8)).astype(int)  # col 0 = left (+az), like TOF_AZ
    dist, status = np.zeros((8, 8)), np.full((8, 8), 254)
    for k in range(8):
        sel = level & (col == k)
        if sel.sum() >= 5:
            dist[3:5, k], status[3:5, k] = np.percentile(horiz[sel], 20) * 1000, 5
    return dist.ravel(), status.ravel()


def objects(img, d, P, f, yaw):
    K, Kd = kmat(f["K"]), kmat(f["K_depth"])
    sx = f["depth_wh"][0] / f["rgb_wh"][0]
    out = []
    for cls, score, (x0, y0, x1, y1) in upright_detect(img, P[:3, :3].T @ [0, 0, 1]):
        ray = to_house_dirs(P, (np.linalg.inv(K) @ [(x0 + x1) / 2, (y0 + y1) / 2, 1.0])[None])[0]
        win = d[int(y0 * sx + (y1 - y0) * sx / 4):int(y1 * sx - (y1 - y0) * sx / 4), int(x0 * sx + (x1 - x0) * sx / 4):int(x1 * sx - (x1 - x0) * sx / 4)]
        rng = None
        if (win > 0).sum() >= 5:
            z = float(np.median(win[win > 0]))
            p = to_house_dirs(P, (np.linalg.inv(Kd) @ [(x0 + x1) / 2 * sx, (y0 + y1) / 2 * sx, 1.0])[None] * z)[0]
            rng = float(np.hypot(p[0], p[1]))
        out.append({"cls": cls, "bearing": float(_wrap(math.atan2(ray[1], ray[0]) - yaw)), "range": rng})
    return out


def hfov(f, P):
    up = P[:3, :3].T @ [0, 0, 1]  # portrait (world up along the image x axis): the short side is horizontal
    K = f["K"]
    return 2 * math.atan(f["rgb_wh"][1] / 2 / K[1]) if abs(up[0]) > abs(up[1]) else 2 * math.atan(f["rgb_wh"][0] / 2 / K[0])


truth = []
for f in frames:
    _, P = ex.load(f)
    p, yaw = pose(P)
    truth.append((p, yaw))
low = [ALL or p[2] < CFG["landmarks"]["low_h"] for p, _ in truth]
first = low.index(True) if any(low) else None
if first is None:
    sys.exit(f"no frame below {CFG['landmarks']['low_h']} m in {ex.session}: record a duck-height pass (or use --all-frames)")
eye = float(np.median([p[2] for (p, _), lw in zip(truth, low) if lw]))
occ = HOUSE / house["occupancy"]["file"] if "occupancy" in house else None
hmap = HouseMap(house, eye_h=eye, occupancy=occ)  # a level ray at this height passes over anything lower
p0, y0_ = truth[first]
filters = {"ToF": ParticleFilter.anywhere(hmap, seed=1), "ToF + objects": ParticleFilter.anywhere(hmap, seed=1),
           "ToF + objects + room colour": ParticleFilter.anywhere(hmap, seed=1),
           "ToF + objects, known start": ParticleFilter(hmap, p0[0], p0[1], y0_, seed=1)}
rng = np.random.default_rng(7)
log = {k: [] for k in filters}
rooms = {k: [] for k in filters}
times, seen = [], 0
for k in range(first, len(frames)):
    (pa, ya), (pb, yb) = truth[max(k - 1, first)], truth[k]
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    fwd, left, turn = dx * math.cos(ya) + dy * math.sin(ya), -dx * math.sin(ya) + dy * math.cos(ya), float(_wrap(yb - ya))
    d_ = math.hypot(fwd, left)
    if ODO == "perfect":
        of, ol, ot = fwd, left, turn
    elif ODO == "gyro":  # a hand-held phone sweeps thousands of degrees: a turn-proportional bias is a duck's, not a gyro's
        of, ol = fwd * 1.04 + rng.normal(0, 0.02 * d_), left + 0.02 * d_ + rng.normal(0, 0.03 * d_)
        ot = turn + rng.normal(0, 0.005 * abs(turn) + 0.01 * d_)
    else:
        of, ol = fwd * 1.04 + rng.normal(0, 0.02 * d_), left + 0.02 * d_ + rng.normal(0, 0.03 * d_)
        ot = turn * 1.03 + 0.05 * d_ + rng.normal(0, 0.01 * abs(turn) + 0.01 * d_)
    for flt in filters.values():
        flt.predict(of, ol, ot)
    if not low[k]:
        continue
    f = frames[k]
    d, P = ex.load(f, conf=1)
    img = ex.rgb(f)
    P3 = cv.depthTo3d(d, kmat(f["K_depth"]))
    dist, status = tof(P3, d, P, pb, yb)
    objs = objects(img, d, P, f, yb)
    seen += len(objs)
    hist = colour_hist(img)
    for name, flt in filters.items():
        flt.update_tof(dist, status)
        if "objects" in name:
            flt.update_landmarks(objs, hfov(f, P))
        if "colour" in name:
            flt.update_room(hist)
        e = flt.estimate()
        log[name].append(math.hypot(e[0] - pb[0], e[1] - pb[1]))
        rooms[name].append(hmap.room(e[0], e[1]) == hmap.room(pb[0], pb[1]))
    times.append(f["t"])

t = np.array(times) - times[0]
res = {"session": ex.session, "query_frames": len(t), "eye_height_m": round(eye, 2), "detections": seen,
       "landmarks_in_map": len(hmap.landmarks), "note": "map built from the same scan: optimistic"}
for name, e in log.items():
    e = np.array(e)
    ok = np.convolve(e < 0.3, np.ones(4), "valid") == 4  # 4 query frames in a row within 0.3 m
    lock = int(np.argmax(ok)) if ok.any() else None
    res[name] = {"median_m": round(float(np.median(e)), 3), "p90_m": round(float(np.percentile(e, 90)), 3),
                 "lock_s": round(float(t[lock]), 1) if lock is not None else None,
                 "after_lock_median_m": round(float(np.median(e[lock:])), 3) if lock is not None else None,
                 "within_0.3m_pct": round(100 * float(np.mean(e < 0.3)), 1), "right_room_pct": round(100 * float(np.mean(rooms[name])), 1)}
(OUT / "results.json").write_text(json.dumps(res, indent=1))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(10, 4.5))
for name, e in log.items():
    ax.plot(t, e, ".-", ms=3, label=f'{name} (median {res[name]["median_m"]:.2f} m)')
ax.set_ylim(0, 4); ax.set_xlabel("time in the scan (s), duck-height frames only"); ax.set_ylabel("position error (m)")
ax.set_title(f"Real images, no stickers, unknown start — {ex.session}"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "lowview.png", dpi=110)
print(json.dumps(res, indent=1))
