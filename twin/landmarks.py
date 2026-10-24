"""Landmarks: things the duck can recognise to find itself. The object detector on the scan's keyframes → the main
surface of each box in 3D → merged across views → candidates; the user's tags name or add candidates; twin/curate.py
decides which to keep and how far to trust them; finish() turns that into house.json landmarks."""
import sys

import cv2 as cv
import numpy as np

from twin import CFG, ROOT
from twin.fuse import vkey
from twin.scan_io import clean_tags

sys.path[:0] = [str(ROOT), str(ROOT / "service")]
from agent.perceive import colour_hist  # noqa: E402
from v0.scan import LANDMARK, detect  # noqa: E402

LM = CFG["landmarks"]


def _rotation(up):
    """cv.rotate code that turns the sensor's landscape picture so world-up is image-up (None: already upright)."""
    ax = int(np.argmax(np.abs(up[:2])))
    if ax == 1 and up[1] < 0:
        return None
    return {(0, False): cv.ROTATE_90_CLOCKWISE, (0, True): cv.ROTATE_90_COUNTERCLOCKWISE, (1, True): cv.ROTATE_180}[(ax, bool(up[ax] > 0))]


def upright_detect(img, up, name=None):
    h, w = img.shape[:2]
    rot = _rotation(up)
    if rot is None:
        return detect(img, classes=LANDMARK)
    back = {cv.ROTATE_90_CLOCKWISE: lambda x, y: (y, h - x), cv.ROTATE_90_COUNTERCLOCKWISE: lambda x, y: (w - y, x),
            cv.ROTATE_180: lambda x, y: (w - x, h - y)}[rot]
    out = []
    for cls, sc, (x0, y0, x1, y1) in detect(cv.rotate(img, rot), classes=LANDMARK):
        (a, b), (c, d) = back(x0, y0), back(x1, y1)
        out.append((cls, sc, [min(a, c), min(b, d), max(a, c), max(b, d)]))
    return out


def lift(P3, d, box, s, P, sc, walls):
    """One detection → the main surface in its box (floor and room walls removed), in the house frame."""
    x0, y0, x1, y1 = np.array(box) * s
    mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    sl = np.s_[max(int(y0 + my), 0):int(np.ceil(y1 - my)), max(int(x0 + mx), 0):int(np.ceil(x1 - mx))]
    pts = P3[sl][d[sl] > 0][:, :3] @ P[:3, :3].T + P[:3, 3]
    pts[:, :2] = pts[:, :2] @ sc.R2.T
    pts[:, 2] -= sc.z0 + sc.ground(pts[:, :2])
    keep = (pts[:, 2] > 0.05) & (pts[:, 2] < 2.2)
    for ax, pos, lo, hi, t in walls:  # also drops door jambs when the object is seen through a doorway
        keep &= ~((np.abs(pts[:, ax] - pos) < t / 2 + 0.06) & (pts[:, 1 - ax] > lo) & (pts[:, 1 - ax] < hi))
    pts = pts[keep]
    if len(pts) < 15:
        return None
    # top-down 4 cm occupancy → connected pieces; keep the biggest (a table in front of a fridge must not win its box)
    cell = np.floor(pts[:, :2] / 0.04).astype(int)
    cell -= cell.min(0)
    occ = np.zeros(cell.max(0)[::-1] + 1, np.uint8)
    occ[cell[:, 1], cell[:, 0]] = 255
    lab = cv.connectedComponents(occ, connectivity=4)[1][cell[:, 1], cell[:, 0]]
    best = pts[lab == np.bincount(lab).argmax()]
    return best if len(best) >= 15 else None


def box3d(pts):
    pts = pts[np.unique(vkey(pts, 0.03), return_index=True)[1]]  # even density: near faces seen often don't dominate
    (cx, cy), _, ang = cv.minAreaRect(pts[:, :2].astype(np.float32))
    ang = np.radians(ang) % (np.pi / 2)
    if min(ang, np.pi / 2 - ang) < np.radians(20):
        ang = 0.0
    c, s = np.cos(ang), np.sin(ang)
    uv = pts[:, :2] @ np.array([[c, -s], [s, c]])
    lo, hi = np.percentile(uv, 2, 0), np.percentile(uv, 98, 0)
    mid = (lo + hi) / 2
    return [float(mid[0] * c - mid[1] * s), float(mid[0] * s + mid[1] * c)], \
        [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(np.percentile(pts[:, 2], 98))], float(ang)


def name_for(label, zone):
    """The detector's word, made sense of in its room (config.toml [names]); None = not a thing worth keeping."""
    label = CFG["names"]["rename"].get(label, label)
    rule = CFG["names"]["only_in"].get(label)
    if rule and not any(w in zone.lower() for w in rule["zones"]):
        return rule["else"] or None
    return label


def zone_at(zones, xy):
    return next((z["label"] for z in zones for r in z["rects"] if r[0] <= xy[0] <= r[2] and r[1] <= xy[1] <= r[3]), "")


def _overlap(a, b):
    lo, hi = np.maximum(a["lo"], b["lo"]), np.minimum(a["hi"], b["hi"])
    inter = np.prod(np.clip(hi - lo, 0, None))
    return inter / max(min(np.prod(a["hi"] - a["lo"]), np.prod(b["hi"] - b["lo"])), 1e-6)


def _boxed(c):
    c["center"], c["size"], c["yaw"] = box3d(np.concatenate(c["pts"]))
    half = np.array(c["size"][:2]) / 2
    c["lo"], c["hi"] = np.array(c["center"]) - half, np.array(c["center"]) + half
    return c


def candidates(ex, frames, sc, zones, walls, detect_fn=upright_detect, every=2):
    """Detector on every 2nd keyframe → [{id, cls, label, pts, views, low_views, best: (frame k, box), source}]."""
    w = sc.wh[0]
    found = []
    for k in range(0, len(frames), every):
        f, P = frames[k], sc.poses[k]
        d, _ = ex.load(f, conf=1)  # grazing table tops are only medium confidence
        dets = detect_fn(ex.rgb(f), P[:3, :3].T @ [0, 0, 1], f["rgb"].rsplit("/", 1)[-1].split(".")[0])
        if dets:
            P3 = cv.depthTo3d(d, sc.Kd)
            for cls, score, box in dets:
                p = lift(P3, d, box, w / f["rgb_wh"][0], P, sc, walls)
                if p is not None:
                    found.append((cls, score, p, k, box))
    groups = []  # same class + touching (shared 15 cm cells) = one object seen from another side
    for cls, score, p, k, box in sorted(found, key=lambda x: -x[1]):
        keys = np.unique(vkey(p, 0.15))
        g = next((g for g in groups if g["cls"] == cls and np.intersect1d(g["keys"], keys).size), None)
        low = int(sc.traj[k, 2] < LM["low_h"])
        if g:
            g["pts"].append(p)
            g["keys"] = np.union1d(g["keys"], keys)
            g["views"] += 1
            g["low_views"] += low
        else:
            groups.append({"cls": cls, "keys": keys, "pts": [p], "views": 1, "low_views": low, "best": (k, box), "source": "detector"})
    out = []
    for g in groups:
        if g["views"] < 2:
            continue
        _boxed(g)
        g["label"] = name_for(g["cls"], zone_at(zones, g["center"]))
        if not g["label"] or max(g["size"][:2]) < 0.2:
            continue
        twin = next((o for o in out if o["label"] == g["label"] and _overlap(o, g) > 0.3), None)
        if twin:  # e.g. "couch" and "bed" (→ sofa outside bedrooms) on the same piece of furniture
            twin["pts"] += g["pts"]
            twin["views"] += g["views"]
            twin["low_views"] += g["low_views"]
            _boxed(twin)
        else:
            out.append(g)
    for i, c in enumerate(out):
        c["id"] = f"c{i}"
        del c["keys"]
    return out


def tagged(ex, frames, sc, walls, cands):
    """The user's tags (crosshair + label): name the candidate under the crosshair, or add the object as a new one."""
    at = {f["i"]: k for k, f in enumerate(frames)}
    w, h = sc.wh
    for t in clean_tags(ex.tags()):
        near_i = min(at, key=lambda i: abs(i - t["i"]), default=None)
        if near_i is None or abs(near_i - t["i"]) > 5:  # its frame is gone (tracking was limited): nothing to aim with
            continue
        k = at[near_i]
        f, P = frames[k], sc.poses[k]
        d, _ = ex.load(f, conf=1)
        u, v = t["uv"]
        for r in (3, 8, 16):
            win = d[max(int(v * h) - r, 0):int(v * h) + r + 1, max(int(u * w) - r, 0):int(u * w) + r + 1]
            if (win > 0).any():
                break
        else:
            continue
        z = float(np.median(win[win > 0]))
        ray = np.linalg.inv(sc.Kd) @ [u * w, v * h, 1.0]
        p = P[:3, :3] @ (ray * z / ray[2]) + P[:3, 3]
        p[:2] = sc.R2 @ p[:2]
        p[2] -= sc.z0 + float(sc.ground(p[:2]))
        label = t["label"].strip().lower()
        near = [c for c in cands if np.all(p[:2] > c["lo"] - 0.3) and np.all(p[:2] < c["hi"] + 0.3)]
        if near:
            c = min(near, key=lambda c: np.hypot(*(np.array(c["center"]) - p[:2])))
            if label == "not a landmark":
                c["removed"] = True
            elif c["source"] != "user" or t.get("web"):  # a fix made on the results page wins over the app's tag
                c.update(label=label, source="user")
            continue
        if label == "not a landmark":
            continue
        rw, rh = f["rgb_wh"]
        box = [(u - 0.12) * rw, (v - 0.12) * rh, (u + 0.12) * rw, (v + 0.12) * rh]
        pts = lift(cv.depthTo3d(d, sc.Kd), d, box, w / rw, P, sc, walls)
        if pts is None:
            pts = p + np.random.default_rng(0).uniform(-0.15, 0.15, (30, 3))
        cands.append(_boxed({"id": f"t{len(cands)}", "cls": LM["tag_cls"].get(label, ""), "label": label, "pts": [pts],
                             "views": 1, "low_views": int(sc.traj[k, 2] < LM["low_h"]), "best": (k, box), "source": "user"}))
    return cands


def crop(ex, frames, sc, c, side=256):
    """The candidate's best view, upright, a little wider than its box: what the curate model looks at."""
    k, (x0, y0, x1, y1) = c["best"]
    img = ex.rgb(frames[k])
    mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    img = img[max(int(y0 - my), 0):int(y1 + my), max(int(x0 - mx), 0):int(x1 + mx)]
    rot = _rotation(sc.poses[k][:3, :3].T @ [0, 0, 1])
    img = img if rot is None else cv.rotate(img, rot)
    s = side / max(img.shape[:2])
    return cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img


def finish(cands, decisions):
    """Apply the keep / label / merge / mobility decisions → house.json landmarks (pre-origin frame)."""
    by = {c["id"]: c for c in cands}
    def root(i, seen=()):
        j = decisions[i].get("merge_into")
        return root(j, (*seen, i)) if isinstance(j, str) and j in by and j != i and j not in seen else i
    for c in cands:
        r = root(c["id"])
        if r != c["id"]:
            by[r]["pts"] += c["pts"]
            by[r]["views"] += c["views"]
            by[r]["low_views"] += c["low_views"]
            c["merged"] = True
    out = []
    for c in cands:
        d = decisions[c["id"]]
        if c.get("merged") or not d["keep"] or d["mobility"] not in LM["weight"]:
            continue
        _boxed(c)
        out.append({"label": c["label"] if c["source"] == "user" else d["label"], "cls": c["cls"], "center": c["center"],
                    "size": c["size"], "yaw": c["yaw"], "views": c["views"], "low_views": c["low_views"], "source": c["source"],
                    "mobility": d["mobility"], "weight": LM["weight"][d["mobility"]], "why": d.get("reason", "")})
    return out


def room_colours(ex, frames, sc, n=30):
    """Each zone's colour fingerprint (hue × saturation), from duck-height frames when the scan has enough of them."""
    out = {}
    for name in sc.names:
        ks = [k for k, f in enumerate(frames) if f["room"] == name]
        low = [k for k in ks if sc.traj[k, 2] < LM["low_h"]]
        ks = low if len(low) >= 5 else ks
        hist = np.mean([colour_hist(ex.rgb(frames[k])) for k in ks[::max(1, len(ks) // n)]], 0)
        out[name] = [round(float(x), 5) for x in hist / hist.sum()]
    return out
