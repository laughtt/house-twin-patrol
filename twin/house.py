"""Assemble house.json (PLAN §6.3, version 2) from the stages, and draw its top view."""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from twin import CFG
from twin.landmarks import zone_at


def seg(w):
    ax, pos, lo, hi = w[:4]
    return ([pos, lo], [pos, hi]) if ax == 0 else ([lo, pos], [hi, pos])


def scan_markers(ex, frames, sc, zones):
    """ArUco markers seen in the scan (optional: a zero-sticker home has none)."""
    from twin.markers_from_scan import find_markers
    out = []
    for m in find_markers(ex, frames, sc.poses):
        c, n = np.array(m["center"], float), np.array(m["normal"], float)
        c[2] -= sc.z0
        c[:2], n[:2] = c[:2] @ sc.R2.T, n[:2] @ sc.R2.T
        out.append({"id": m["id"], "x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "yaw": float(np.arctan2(n[1], n[0])),
                    "size_m": m["size_m"], "views": m["views"], "room": zone_at(zones, c[:2])})
    return out


def hand_markers(path, o, walls_full, objects):
    """Placed by hand in the house frame (origin applied), snapped onto the face of the wall or furniture behind them
    when one is within 30 cm: a marker is never buried in a counter or a wall."""
    out = []
    for m in json.loads(Path(path).read_text())["markers"]:
        x, y = m["x"] + o[0], m["y"] + o[1]
        n = np.array([np.cos(m["yaw"]), np.sin(m["yaw"])])
        ax = 0 if abs(n[0]) > abs(n[1]) else 1  # facing ±x: on a wall along y (axis 0)
        along, across = (y, x) if ax == 0 else (x, y)
        faces = [w_[1] + (w_[4] / 2 if w_[5] else 0) * np.sign(n[ax]) for w_ in walls_full  # outer walls: line = face
                 if w_[0] == ax and w_[2] - 0.1 <= along <= w_[3] + 0.1]
        for ob in objects:
            c, s = ob["center"], ob["size"]
            if abs(along - c[1 - ax]) <= s[1 - ax] / 2:
                faces.append(c[ax] + s[ax] / 2 * np.sign(n[ax]))
        near = [f for f in faces if abs(f - across) <= 0.3]
        if near:
            face = max(near, key=lambda f: f * np.sign(n[ax]))
            x, y = (face, y) if ax == 0 else (x, face)
        out.append({**m, "x": float(x), "y": float(y)})
    return out


def assemble(session, sc, spaces, zones, walls_full, doors, objects, landmarks, markers, colours, o):
    r3 = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    shift = lambda p: r3(np.subtract(p, o))  # noqa: E731
    ceiling = float(np.percentile(sc.V[:, 2], 99.5))
    return {
        "version": 2, "units": "m", "source": f"lidar/{session}",
        "rooms": [{**z, "polygon": [shift(p) for p in z["polygon"]], "rects": [shift(r[:2]) + shift(r[2:]) for r in z["rects"]],
                   **({"colour_hs": colours[z["label"]]} if z["label"] in colours else {})} for z in zones],
        "spaces": [{"name": s["name"], "polygon": [shift(p) for p in s["polygon"]]} for s in spaces],
        "walls": [{"a": shift(a), "b": shift(b), "h": round(min(ceiling, 3.0), 2) if ceiling > 2.0 else CFG["house"]["wall_h"],
                   "t": round(w_[4], 3), "shared": w_[5]} for w_ in walls_full for a, b in [seg(w_)]],
        "doors": [{**d_, "center": shift(d_["center"]), "threshold_h": 0.015} for d_ in doors],
        "objects": [{**ob, "center": shift(ob["center"])} for ob in objects],
        "landmarks": [{**lm, "center": shift(lm["center"]), "size": r3(lm["size"]), "yaw": round(lm["yaw"], 4)} for lm in landmarks],
        "markers": [{**m, "x": round(m["x"] - o[0], 3), "y": round(m["y"] - o[1], 3), "z": round(m["z"], 3), "yaw": round(m["yaw"], 4)}
                    for m in markers],
        "accuracy": {"wall_len_err_pct_median": None, "n_checked": 0},
    }


def occupancy(path, V, o, gm, cell=0.05, dz=0.1, top=2.0):
    """Where the LiDAR saw any surface, in 5 cm cells and 10 cm height bands up to 2 m (house frame): the real shape
    of walls and furniture, for checking a range sensor at any height. Saved next to house.json."""
    p = V[(V[:, 2] > 0.05) & (V[:, 2] < top)]
    lo = p[:, :2].min(0) - o
    ij = np.floor((p[:, :2] - o - lo) / cell).astype(int)
    k = np.floor(p[:, 2] / dz).astype(int)
    occ = np.zeros((int(top / dz), ij[:, 1].max() + 1, ij[:, 0].max() + 1), bool)
    occ[k, ij[:, 1], ij[:, 0]] = True
    np.savez_compressed(path, occ=occ, lo=lo, cell=cell, dz=dz, ground=gm[2], ground_lo=gm[0] - o, ground_tile=gm[1])  # + floor heights
    stage_pictures(Path(path).parent, occ, gm[2])
    return {"file": Path(path).name, "cell": cell, "dz": dz, "lo": [round(float(v), 3) for v in lo], "shape": list(occ.shape)}


def stage_pictures(out, occ, ground, px=3):
    """What two stages saw, for the results page: the local floor tiles and what a 20 cm sensor can hit (north up)."""
    seen = cv.resize(occ.any(0).astype(np.uint8) * 70, None, fx=px, fy=px, interpolation=cv.INTER_NEAREST)
    duck = cv.resize(occ[1:4].any(0).astype(np.uint8) * 255, None, fx=px, fy=px, interpolation=cv.INTER_NEAREST)
    img = cv.merge([np.maximum(seen, duck // 3), np.maximum(seen, duck), np.maximum(seen, duck)])  # grey: any height, cyan: 10-40 cm
    cv.imwrite(str(out / "stage_occupancy.png"), img[::-1])
    mid = float(np.median(ground))
    norm = np.clip((ground - mid + 0.15) / 0.3 * 255, 0, 255).astype(np.uint8)  # ±15 cm around the usual floor
    tiles = cv.applyColorMap(cv.resize(norm, None, fx=20, fy=20, interpolation=cv.INTER_NEAREST), cv.COLORMAP_VIRIDIS)[::-1].copy()
    cv.putText(tiles, f"floor {ground.min() - mid:+.2f} .. {ground.max() - mid:+.2f} m around the usual level", (8, 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.imwrite(str(out / "stage_floor.png"), tiles)


def top_view(path, house, wall_pts, traj, tlab, px=80):
    allp = np.concatenate([np.array(r["polygon"]) for r in house["spaces"]])
    lo = allp.min(0) - 0.6
    size = np.ceil((allp.max(0) + 0.6 - lo) * px).astype(int)
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    to = lambda p: (int((p[0] - lo[0]) * px), int(size[1] - (p[1] - lo[1]) * px))  # noqa: E731
    colors = [(230, 200, 160), (170, 220, 170), (160, 200, 240), (210, 170, 220), (160, 230, 230), (200, 200, 200)]
    for i, r in enumerate(house["rooms"]):
        cv.fillPoly(img, [np.array([to(p) for p in r["polygon"]], np.int32)], colors[i % len(colors)])
    for p in wall_pts[::3]:
        q = to(p)
        if 0 <= q[0] < size[0] and 0 <= q[1] < size[1]:
            img[q[1], q[0]] = (120, 120, 120)
    for wl in house["walls"]:
        cv.line(img, to(wl["a"]), to(wl["b"]), (40, 40, 40), max(2, int(wl["t"] * px)))
    for d in house["doors"]:
        u = np.array([0.0, 1.0]) if d["axis"] == 0 else np.array([1.0, 0.0])
        cv.line(img, to(d["center"] - u * d["width"] / 2), to(d["center"] + u * d["width"] / 2), (60, 200, 60), 8)
    for o in house["objects"]:
        box = cv.boxPoints((tuple(o["center"]), tuple(o["size"][:2]), np.degrees(o["yaw"])))
        cv.polylines(img, [np.array([to(p) for p in box], np.int32)], True, (40, 40, 200), 2)
        cv.putText(img, o["label"], to(np.add(o["center"], [-0.3, 0])), cv.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 200), 1)
    for lm in house["landmarks"]:  # star size = trust; filled = tagged by the user
        r = int(4 + 6 * lm.get("weight", 0.5))
        cv.drawMarker(img, to(lm["center"]), (0, 140, 255), cv.MARKER_STAR, 2 * r, 3 if lm.get("source") == "user" else 1)
        cv.putText(img, lm["label"], to(np.add(lm["center"], [0.08, 0.08])), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 110, 220), 1)
    for i in range(1, len(traj)):
        low = traj[i, 2] < CFG["landmarks"]["low_h"]
        cv.line(img, to(traj[i - 1]), to(traj[i]), (0, 0, 0) if low else [c * 0.6 for c in colors[tlab[i] % len(colors)]], 4 if low else 2)
    cv.circle(img, to(traj[0]), 6, (200, 80, 30), -1)
    for r in house["rooms"]:
        p = np.array(r["polygon"])
        w, l = np.ptp(p, 0)
        cv.putText(img, f"{r['name']} {w:.2f}x{l:.2f}m", to(p.mean(0) - [0.8, 0]), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 0), 2)
    cv.circle(img, to((0, 0)), 5, (0, 0, 255), -1)
    cv.imwrite(str(path), img)
