"""Synthetic LiDAR walk through a two-room house, written exactly like the iPhone app's export, then built.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/synth_lidar.py
--house multi: open-plan L (kitchen + living), hallway, bedroom, a glass door, ArUco markers → synth-multi.zip and
synth-multi.truth.json (export + ground truth only, no build).
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lidar" / "pipeline"))
import build_house  # noqa: E402

OUT = Path(__file__).parent.parent / "out" / "lidar"
SESSION = "lidar-synth"
# house frame: z up, metres. Living room [0,4.6]x[0,3.8], 12 cm wall, kitchen [4.72,8.02]x[0,3.1]
ROOMS = {"Living room": (0, 0, 4.6, 3.8), "Kitchen": (4.72, 0, 8.02, 3.1)}
DOOR = (4.66, 1.45, 0.9)  # centre x, y and width, in the shared wall
SOLIDS = [((-1, -1, -0.2), (9, 5, 0)), ((-1, -1, 2.5), (9, 5, 2.7)),  # floor, ceiling
          ((-0.2, -1, 0), (0, 5, 2.5)), ((-1, -0.2, 0), (9, 0, 2.5)), ((-1, 3.8, 0), (4.72, 5, 2.5)),
          ((4.6, 3.1, 0), (9, 5, 2.5)), ((8.02, -1, 0), (9, 5, 2.5)),
          ((4.6, -1, 0), (4.72, 1.0, 2.5)), ((4.6, 1.9, 0), (4.72, 5, 2.5)), ((4.6, 1.0, 2.05), (4.72, 1.9, 2.5))]
OBJECTS = {"couch": ((0.3, 2.9, 0), (2.3, 3.75, 0.85)), "dining table": ((6.0, 1.2, 0), (7.2, 2.0, 0.75)),
           "refrigerator": ((7.3, 2.3, 0), (7.97, 3.05, 1.8))}
BOXES = SOLIDS + list(OBJECTS.values())
GLASS, MARKERS = set(), []  # indices in BOXES that the LiDAR sees through; (id, centre, yaw of the facing)
LABEL = lambda p, prev: "Kitchen" if p[0] > DOOR[0] + 0.3 else "Living room"  # noqa: E731  taps a step after the door
RGB_WH, DEPTH_WH, F, EYE = (960, 720), (256, 192), 725.0, 1.35
PATH = [(1.0, 1.0), (3.6, 1.0), (3.6, 2.4), (1.0, 2.2), (1.0, 1.4), (3.9, 1.45), (5.4, 1.45), (5.4, 0.6),
        (7.6, 0.6), (7.6, 1.8), (5.4, 2.6), (5.4, 1.45)]
YAW0, rng = 0.4, np.random.default_rng(0)
# z-up house → ARKit world (y up): x_ar = x, y_ar = z, z_ar = -y, then a turn and the start pose as origin
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # z-up from ARKit
RZ = np.array([[np.cos(YAW0), -np.sin(YAW0), 0], [np.sin(YAW0), np.cos(YAW0), 0], [0, 0, 1]])


def walk(step=0.15):
    pts, labels, turn = [], [], []
    for a, b in zip(PATH[:-1], PATH[1:]):
        a, b, spin = np.array(a[:2]), np.array(b[:2]), len(b) > 2
        n = max(1, int(np.hypot(*(b - a)) / step))
        for s in np.arange(n) / n:
            p = a + (b - a) * s
            pts.append(p)
            labels.append(LABEL(p, labels[-1] if labels else None))
        turn += [0.0] * n
        if spin:  # turn once on the spot so the camera sweeps every wall
            pts += [b] * 24
            labels += [LABEL(b, labels[-1])] * 24
            turn += list(np.arange(1, 25) * np.pi / 12)
    pts, turn = np.array(pts), np.array(turn)
    g = np.gradient(pts, axis=0)
    head = np.arctan2(g[:, 1], g[:, 0])
    for i in np.nonzero(np.hypot(*g.T) < 1e-9)[0]:
        head[i] = head[i - 1]
    sweep = np.radians(55) * np.sin(np.arange(len(pts)) * 0.35) * (turn == 0)  # sweep the walls while walking
    return pts, np.unwrap(head) + turn + sweep, labels


def pose(xy, yaw, pitch=np.radians(15)):
    """OpenCV camera in the house frame, phone held in portrait: image x points down, y left, z forward."""
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0])
    return np.c_[np.cross(left, fwd), left, fwd], np.array([xy[0], xy[1], EYE])


def render(R, c, wh, f, glass=True):
    u, v = np.meshgrid(np.arange(wh[0]) + 0.5, np.arange(wh[1]) + 0.5)
    d = np.stack([(u - wh[0] / 2) / f, (v - wh[1] / 2) / f, np.ones_like(u)], -1) @ R.T
    t, sid, ax = np.full(u.shape, np.inf), np.zeros(u.shape, np.int64), np.zeros(u.shape, np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, (lo, hi) in enumerate(BOXES):
            if i in GLASS and not glass:
                continue
            t1, t2 = (np.array(lo) - c) / d, (np.array(hi) - c) / d
            tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            m = (tn < tf) & (tn > 0) & (tn < t)
            t[m], sid[m], ax[m] = tn[m], i, np.minimum(t1, t2).argmax(-1)[m]
    p = c + d * np.minimum(t, 1e3)[..., None]  # rays through the glass may hit nothing
    k = np.floor(p / 0.15).astype(np.int64)
    h = (k[..., 0] * 73856093) ^ (k[..., 1] * 19349663) ^ (k[..., 2] * 83492791) ^ (sid * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    img = (np.stack([h & 255, (h >> 8) & 255, (h >> 16) & 255], -1) // 2 + 60).astype(np.uint8)
    img[np.isin(sid, list(GLASS))] = (235, 232, 226)  # daylight through the window
    for mid, mc, yaw in MARKERS:  # printed marker on the wall: right, down, out of the wall
        n = np.array([np.cos(yaw), np.sin(yaw), 0])
        q = (p - mc) @ np.c_[np.cross(-n, [0, 0, 1]), [0, 0, -1], n]
        m = (np.abs(q[..., :2]) < 0.1).all(-1) & (np.abs(q[..., 2]) < 0.01)
        ij = ((q[m, :2] + 0.1) / 0.025).astype(int).clip(0, 7)
        img[m] = TEX[mid][ij[:, 1], ij[:, 0], None]
    cos = np.abs(np.take_along_axis(d, ax[..., None], -1)[..., 0]) / np.linalg.norm(d, axis=-1)
    return img, t.astype(np.float32), cos, sid  # t is z-depth because camera rays have z = 1


def box_px(R, c, lo, hi, sid, k):
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pc = (pts - c) @ R
    if (pc[:, 2] <= 0.2).any():
        return None
    uv = pc[:, :2] / pc[:, 2:] * F + np.array(RGB_WH) / 2
    x0, y0 = np.maximum(uv.min(0), 0)
    x1, y1 = np.minimum(uv.max(0), RGB_WH)
    if x1 - x0 < 20 or y1 - y0 < 20 or (sid[int(y0):int(y1), int(x0):int(x1)] == k).mean() < 0.3:
        return None  # out of view or hidden behind a wall
    return [x0, y0, x1, y1]


def export():
    root = OUT / "_synth" / SESSION
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("rgb", "depth", "conf"):
        (root / sub).mkdir(parents=True)
    pts, yaws, labels = walk()
    t0 = -RZ @ np.array([*pts[0], EYE])
    fd = F * DEPTH_WH[0] / RGB_WH[0]
    lines, events, truth = [], [], {}
    for i, (xy, yaw, room) in enumerate(zip(pts, yaws, labels)):
        R, c = pose(xy, yaw)
        name = f"{i:06d}"
        img, _, _, sid = render(R, c, RGB_WH, F)
        cv.imwrite(str(root / f"rgb/{name}.jpg"), cv.GaussianBlur(img, (0, 0), 0.7), [cv.IMWRITE_JPEG_QUALITY, 60])
        _, z, cos, _ = render(R, c, DEPTH_WH, fd, glass=False)
        z = z * (1 + 0.004 * rng.standard_normal(z.shape)) + 0.003 * rng.standard_normal(z.shape)
        conf = np.where(z > 5, 0, np.where((z < 3.5) & (cos > 0.3), 2, 1)).astype(np.uint8)
        (root / f"depth/{name}.bin").write_bytes(np.minimum(z, 9).astype(np.float16).tobytes())
        (root / f"conf/{name}.bin").write_bytes(conf.tobytes())
        jitter = cv.Rodrigues(rng.normal(0, np.radians(0.15), 3))[0]
        Rar = M.T @ RZ @ jitter @ R @ np.diag([1, -1, -1])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rar, M.T @ (RZ @ c + t0) + rng.normal(0, 0.004, 3)
        K = lambda s: [F * s, F * s, RGB_WH[0] / 2 * s, RGB_WH[1] / 2 * s]  # noqa: E731
        lines.append({"i": i, "t": i / 4, "rgb": f"rgb/{name}.jpg", "depth": f"depth/{name}.bin",
                      "conf": f"conf/{name}.bin", "rgb_wh": list(RGB_WH), "K": K(1), "depth_wh": list(DEPTH_WH),
                      "K_depth": K(DEPTH_WH[0] / RGB_WH[0]), "T": T.ravel().tolist(), "tracking": "normal",
                      "room": room})
        if i and labels[i - 1] != room:
            events.append({"t": i / 4, "i": i, "event": "room", "room": room, "T": T.ravel().tolist()})
        truth[name] = [(cls, b) for k, (cls, (lo, hi)) in enumerate(OBJECTS.items())
                       if (b := box_px(R, c, lo, hi, sid, len(SOLIDS) + k))]
    (root / "frames.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (root / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in events))
    (root / "session.json").write_text(json.dumps({"session": SESSION, "device": "synthetic", "fps": 4}))
    zpath = OUT / f"{SESSION}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:  # same layout as the app's zip: <session>/...
        for f in sorted(root.rglob("*")):
            z.write(f, f.relative_to(root.parent))
    shutil.rmtree(OUT / "_synth")
    return zpath, truth, np.r_[np.c_[RZ, t0], [[0, 0, 0, 1]]]  # house → the z-up world of Export.load


def multi():
    """Open-plan L (kitchen + living) north-west, hallway and bedroom behind walls with doors, glass door north."""
    global SESSION, ROOMS, SOLIDS, OBJECTS, BOXES, PATH, GLASS, MARKERS, LABEL, TEX, DOORS, WALLS
    SESSION = "synth-multi"
    ROOMS = {"Kitchen": [[0, 0], [3.6, 0], [3.6, 3.6], [0, 3.6]],
             "Living room": [[0, 3.6], [3.6, 3.6], [3.6, 3.72], [8, 3.72], [8, 7], [0, 7]],
             "Hallway": [[3.72, 1], [5, 1], [5, 3.6], [3.72, 3.6]],
             "Bedroom": [[5.12, 1], [8, 1], [8, 3.6], [5.12, 3.6]]}
    DOORS = [((4.36, 3.66), 0.85), ((5.06, 2.3), 0.85)]
    WALLS = [((-0.2, -0.2), (0, 7.2)), ((-0.2, -0.2), (3.8, 0)), ((3.6, -0.2), (3.8, 1)), ((3.6, 0.8), (8.2, 1)),
             ((8, 0.8), (8.2, 7.2)), ((-0.2, 7), (1.2, 7.2)), ((3.6, 7), (8.2, 7.2)),  # outer
             ((3.6, 1), (3.72, 3.72)), ((3.6, 3.6), (3.935, 3.72)), ((4.785, 3.6), (8, 3.72)),  # kitchen|hall, door 1
             ((5, 1), (5.12, 1.875)), ((5, 2.725), (5.12, 3.6))]  # hall|bedroom, door 2
    glass = ((1.2, 7, 0), (3.6, 7.2, 2.2))
    SOLIDS = [((-0.2, -0.2, -0.2), (8.2, 7.2, 0)), ((-0.2, -0.2, 2.5), (8.2, 7.2, 2.7)), glass,
              ((1.2, 7, 2.2), (3.6, 7.2, 2.5)), ((3.935, 3.6, 2.05), (4.785, 3.72, 2.5)),
              ((5, 1.875, 2.05), (5.12, 2.725, 2.5)), ((-1, 9.5, 0), (9, 9.55, 1.2))]  # headers, garden fence
    SOLIDS += [((*a, 0), (*b, 2.5)) for a, b in WALLS]
    OBJECTS = {"counter": ((0, 0.6, 0), (0.6, 2.8, 0.9)), "refrigerator": ((2.85, 0.05, 0), (3.55, 0.75, 1.8)),
               "dining table": ((1.0, 4.6, 0), (2.2, 5.4, 0.75)), "couch": ((4.7, 6.1, 0), (6.9, 6.95, 0.85)),
               "bed": ((6.2, 1.6, 0), (7.95, 3.55, 0.55))}
    BOXES, GLASS = SOLIDS + list(OBJECTS.values()), {2}
    MARKERS = [(0, (0.6, 1.7, 0.2), 0), (2, (4.1, 7, 0.2), -np.pi / 2), (3, (4.36, 1, 0.2), np.pi / 2),
               (5, (8, 1.3, 0.2), np.pi)]
    aruco = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)  # 15 cm black square + one white cell
    TEX = {m[0]: cv.copyMakeBorder(cv.aruco.generateImageMarker(aruco, m[0], 6), 1, 1, 1, 1, cv.BORDER_CONSTANT,
                                   value=255) // 4 * 3 + 30 for m in MARKERS}
    PATH = [(2.0, 1.0), (2.4, 2.0, "turn"), (2.8, 1.7), (1.8, 1.7), (2.0, 3.0), (2.9, 4.3), (2.6, 6.0),
            (3.6, 5.3, "turn"), (6.5, 5.0, "turn"), (7.2, 4.3), (4.36, 4.4), (4.36, 2.6, "turn"), (4.36, 2.3),
            (5.7, 2.3), (5.7, 1.6, "turn"), (7.2, 1.3), (5.6, 1.6)]
    LABEL = lambda p, prev: next((k for k, v in ROOMS.items()  # noqa: E731  a step inside a zone taps it
                                  if (np.min(v, 0) + 0.3 <= p).all() and (p <= np.max(v, 0) - 0.3).all()), prev)


def write_truth(H, boxes2d):
    walls = []
    for lo, hi in WALLS:  # centre line along the long side, thickness across
        ax, m = int(hi[0] - lo[0] < hi[1] - lo[1]), np.add(lo, hi) / 2
        a, b = m.copy(), m.copy()
        a[ax], b[ax] = lo[ax], hi[ax]
        walls.append({"a": a.round(3).tolist(), "b": b.round(3).tolist(), "t": round(hi[1 - ax] - lo[1 - ax], 3)})
    lo, hi = BOXES[2][0], BOXES[2][1]
    truth = {"frame": "z-up metres; house_to_world maps it to the z-up world of build_house.Export.load poses",
             "house_to_world": H.round(6).tolist(),
             "rooms": [{"name": k, "polygon": v} for k, v in ROOMS.items()],
             "outer_polygon": [[0, 0], [3.6, 0], [3.6, 1], [8, 1], [8, 7], [0, 7]],
             "walls": walls,
             "doors": [{"center": list(c), "width": w} for c, w in DOORS],
             "glass": [{"a": [lo[0], lo[1]], "b": [hi[0], lo[1]], "h": hi[2]}],
             "markers": [{"id": i, "center": list(c), "yaw": round(float(y), 6), "size_m": 0.15} for i, c, y in MARKERS],
             "objects": [{"label": k, "center": ((np.add(a, b)) / 2).tolist(), "size": np.subtract(b, a).round(3).tolist()}
                         for k, (a, b) in OBJECTS.items()],
             "boxes2d": boxes2d}  # per rgb frame: [label, [x0, y0, x1, y1]] of visible objects, for a fake detector
    path = OUT / f"{SESSION}.truth.json"
    path.write_text(json.dumps(truth, indent=1))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", choices=["two", "multi"], default="two")
    if ap.parse_args().house == "multi":
        multi()
    zpath, truth, H = export()
    print(f"export {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    if SESSION == "synth-multi":
        sys.exit(print(f"truth {write_truth(H, truth)}"))
    # the detector can't name hashed boxes, so feed it the true 2D boxes: this tests the 3D lifting + merging
    fake = lambda img, up, name: [(cls, 0.9, b) for cls, b in truth[name]]  # noqa: E731
    house, out = build_house.build(zpath, detect_fn=fake)
    print(f"output {out}")
    true = {build_house.slug(k): (v[2] - v[0], v[3] - v[1]) for k, v in ROOMS.items()}
    for r in house["rooms"]:
        p = np.array(r["polygon"])
        w, l = np.ptp(p, 0)
        tw, tl = true[r["name"]]
        if abs(w - tl) + abs(l - tw) < abs(w - tw) + abs(l - tl):
            tw, tl = tl, tw
        print(f"{r['name']:12s} {w:.3f} x {l:.3f} m  (true {tw:.2f} x {tl:.2f})  "
              f"err {100 * (w / tw - 1):+.1f} % / {100 * (l / tl - 1):+.1f} %")
    for d in house["doors"]:
        print(f"door width {d['width']:.2f} m (true {DOOR[2]})  centre {d['center']} (true {list(DOOR[:2])})")
    for o in house["objects"]:
        lo, hi = np.array(OBJECTS.get(o["label"], ((0, 0, 0), (0, 0, 0))))
        print(f"{o['label']:13s} size {o['size']} centre {o['center']}  (true {(hi - lo).round(2).tolist()} "
              f"centre {((lo + hi) / 2)[:2].round(2).tolist()})")
