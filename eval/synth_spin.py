"""Synthetic spin in a box room with known sizes: checks v0's lens, floor, wall and height maths.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/synth_spin.py [out.mp4] [phone offset m]
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "service"))
from v0.scan import DEPTH_WH, Scan, depth  # noqa: E402

W, L, H, CAM, HFOV, WH = 4.2, 3.4, 2.5, np.array([1.5, 1.2, 1.35]), 68.0, (960, 540)
BOXES = {"dining table": ([2.4, 1.6, 0], [3.6, 2.4, 0.75]), "refrigerator": ([0.05, 2.6, 0], [0.75, 3.35, 1.8])}
OFFSET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
F = WH[0] / 2 / np.tan(np.radians(HFOV) / 2)


def pose(yaw, pitch=np.radians(18)):  # aimed a little down so the floor line is in view
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    down = np.array([-np.cos(yaw) * np.sin(pitch), -np.sin(yaw) * np.sin(pitch), -np.cos(pitch)])
    R = np.c_[[np.sin(yaw), -np.cos(yaw), 0], down, fwd]  # columns: camera x, y, z in world
    return R, CAM + OFFSET * np.array([np.cos(yaw), np.sin(yaw), 0])  # the phone circles in front of the body


def render(yaw):
    R, c = pose(yaw)
    u, v = np.meshgrid(np.arange(WH[0]) + 0.5, np.arange(WH[1]) + 0.5)
    d = np.stack([(u - WH[0] / 2) / F, (v - WH[1] / 2) / F, np.ones_like(u)], -1) @ R.T
    t, sid = np.full(u.shape, np.inf), np.zeros(u.shape, np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for ax, hi in enumerate((W, L, H)):
            tt = np.where(d[..., ax] > 0, (hi - c[ax]) / d[..., ax], -c[ax] / d[..., ax])
            m = (tt > 0) & (tt < t)
            t[m], sid[m] = tt[m], ax * 2 + (d[..., ax] > 0)[m]
        for i, (lo, hi) in enumerate(BOXES.values()):
            t1, t2 = (np.array(lo) - c) / d, (np.array(hi) - c) / d
            tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            m = (tn < tf) & (tn > 0) & (tn < t)
            t[m], sid[m] = tn[m], 10 + i
    k = np.floor((c + d * t[..., None]) / 0.15).astype(np.int64)
    h = (k[..., 0] * 73856093) ^ (k[..., 1] * 19349663) ^ (k[..., 2] * 83492791) ^ (sid * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    img = (np.stack([h & 255, (h >> 8) & 255, (h >> 16) & 255], -1) // 2 + 60).astype(np.uint8)
    return img, t.astype(np.float32)  # t is z-depth because camera rays have z = 1


def box_px(yaw, lo, hi):
    R, c = pose(yaw)
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pc = (pts - c) @ R
    if (pc[:, 2] <= 0.1).any():
        return None
    uv = pc[:, :2] / pc[:, 2:] * F + np.array(WH) / 2
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    if x0 < 0 or x1 > WH[0] or y0 < 3:
        return None
    return [x0, y0, x1, min(y1, WH[1])]


yaws = np.radians(np.linspace(0, 380, 90))
frames, gt = [], {}
for yaw in yaws:
    img, z = render(yaw)
    frames.append(img)
    gt[id(img)] = cv.resize(z, DEPTH_WH, interpolation=cv.INTER_NEAREST)
if len(sys.argv) > 1:
    vw = cv.VideoWriter(sys.argv[1], cv.VideoWriter_fourcc(*"mp4v"), 4.5, WH)  # the 380° turn takes 20 s
    [vw.write(f) for f in frames]
    vw.release()

for label, fn in [("true depth", lambda im: gt[id(im)]), ("depth model", depth)]:
    s = Scan(frames, fn)
    print(f"\n== {label}: {len(s.frames)}/{len(frames)} frames, spin {s.spin_deg:.0f}° (true 380)")
    print(f"lens HFOV {s.hfov:.1f}° (true {HFOV})  camera height {s.cam_h:.2f} m (true 1.35)  "
          f"frame scales {s.s.min():.2f}-{s.s.max():.2f}")
    print(f"room {s.size[0]:.2f} x {s.size[1]:.2f} m (true {W} x {L}, either order)  walls {s.walls}")
    keep = [next(i for i, f in enumerate(frames) if f is g) for g in s.frames]
    for name, (lo, hi) in BOXES.items():
        tops = [m[0] for k, i in enumerate(keep) if (b := box_px(yaws[i], lo, hi)) and (m := s.measure(k, b))]
        print(f"{name}: top {np.median(tops):.2f} m from {len(tops)} views (true {hi[2]})" if tops
              else f"{name}: top never fully in view")
    print("detector found:", [(o["cls"], round(o["top_model_m"], 2)) for o in s.objects()])
