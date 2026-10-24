"""iPhone LiDAR export (lidar/app) → house.json (PLAN §6.3), top.png, cloud.ply, report.json. OpenCV 5.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python lidar/pipeline/build_house.py <export.zip|folder> [out_dir]

Frames: ARKit world is y up (gravity), camera x right / y up / z back. OpenCV camera is x right / y down / z forward,
so T_cv = T_ar · diag(1,-1,-1,1); the house frame is z up: (x, y, z) = (x_ar, -z_ar, y_ar).
"""
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT))
from v0.scan import detect  # noqa: E402

import tomllib  # noqa: E402
from twin import floorplan  # noqa: E402
CFG = tomllib.loads((ROOT / "config.toml").read_text())

VOX, MAX_D, WALL_T, WALL_H = 0.025, 5.0, 0.10, 1.6
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # ARKit world (y up) → house (z up)
FLIP = np.diag([1.0, -1, -1])  # ARKit camera → OpenCV camera
TABLES = {"dining table", "bench"}


def slug(s):
    return s.strip().lower().replace(" ", "_")


def kmat(k):
    return np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], np.float32)


def vkey(p, vox=VOX):
    k = np.floor(p / vox).astype(np.int64) + (1 << 17)
    return (k[:, 0] << 36) | (k[:, 1] << 18) | k[:, 2]


def means(inv, n, *arrays):
    cnt = np.bincount(inv, minlength=n)
    return cnt, [np.stack([np.bincount(inv, a[:, j], n) for j in range(a.shape[1])], 1) / np.maximum(cnt, 1)[:, None]
                 for a in arrays]


class Export:
    def __init__(self, src):
        src = Path(src)
        if src.suffix == ".zip":
            self.z = zipfile.ZipFile(src)
            self.pre = next(n for n in self.z.namelist() if n.endswith("frames.jsonl"))[:-len("frames.jsonl")]
            self.session = Path(self.pre).name or src.stem
        else:
            self.z, self.pre, self.session = None, f"{src}/", src.name
        self.frames = [json.loads(s) for s in self.read("frames.jsonl").decode().splitlines() if s.strip()]
        try:  # zones fixed in the app's review screen win over the live taps
            fix = {i: r["room"] for r in json.loads(self.read("labels.json"))["ranges"] for i in range(r["from"], r["to"] + 1)}
            for f in self.frames:
                f["room"] = fix.get(f["i"], f["room"])
        except (KeyError, FileNotFoundError):
            pass

    def read(self, name):
        return self.z.read(self.pre + name) if self.z else Path(self.pre + name).read_bytes()

    def load(self, f, conf=2):
        w, h = f["depth_wh"]
        d = np.frombuffer(self.read(f["depth"]), np.float16).reshape(h, w).astype(np.float32)
        c = np.frombuffer(self.read(f["conf"]), np.uint8).reshape(h, w)
        d = np.where((c >= conf) & (d > 0.2) & (d < MAX_D), d, 0).astype(np.float32)
        T = np.array(f["T"]).reshape(4, 4)
        P = np.eye(4)
        P[:3, :3], P[:3, 3] = M @ T[:3, :3] @ FLIP, M @ T[:3, 3]
        return d, P

    def rgb(self, f):
        return cv.imdecode(np.frombuffer(self.read(f["rgb"]), np.uint8), cv.IMREAD_COLOR)


def upright_detect(img, up, name=None):
    """The app saves the sensor's landscape picture; turn it so world-up is image-up for the detector."""
    h, w = img.shape[:2]
    ax = np.argmax(np.abs(up[:2]))
    if ax == 1 and up[1] < 0:
        return detect(img)
    rot, back = {(0, False): (cv.ROTATE_90_CLOCKWISE, lambda x, y: (y, h - x)),  # up = image left
                 (0, True): (cv.ROTATE_90_COUNTERCLOCKWISE, lambda x, y: (w - y, x)),  # up = image right
                 (1, True): (cv.ROTATE_180, lambda x, y: (w - x, h - y))}[(int(ax), bool(up[ax] > 0))]
    out = []
    for cls, sc, (x0, y0, x1, y1) in detect(cv.rotate(img, rot)):
        (a, b), (c, d) = back(x0, y0), back(x1, y1)
        out.append((cls, sc, [min(a, c), min(b, d), max(a, c), max(b, d)]))
    return out


def lowest_layer(z, lo=-3.0, hi=1.0):
    hist, edges = np.histogram(z, bins=np.arange(lo, hi, 0.02))
    hist = np.convolve(hist, np.ones(3), "same")
    z0 = edges[np.nonzero(hist >= 0.15 * hist.max())[0][0]] + 0.01
    return float(np.median(z[np.abs(z - z0) < 0.04]))


def manhattan(n):
    # wall normals' direction mod 90°: histogram mode, then a 4θ circular mean around it
    phi = np.arctan2(n[:, 1], n[:, 0])
    hist = np.bincount((np.degrees(phi) % 90).astype(int) % 90, minlength=90)
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid")
    mode = np.radians(np.argmax(hist))
    near = np.abs((phi - mode + np.pi / 4) % (np.pi / 2) - np.pi / 4) < np.radians(6)
    return float(np.angle(np.exp(4j * phi[near]).sum()) / 4) % (np.pi / 2)


def seg(w):
    ax, pos, lo, hi = w[:4]
    return ([pos, lo], [pos, hi]) if ax == 0 else ([lo, pos], [hi, pos])


def lift(P3, d, box, s, P, z0, R2, walls):
    """One detection → the main surface in its box (floor and room walls removed), in the house frame."""
    x0, y0, x1, y1 = np.array(box) * s
    mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    sl = np.s_[int(y0 + my):int(np.ceil(y1 - my)), int(x0 + mx):int(np.ceil(x1 - mx))]
    pts = P3[sl][d[sl] > 0][:, :3] @ P[:3, :3].T + P[:3, 3]
    pts[:, :2] = pts[:, :2] @ R2.T
    pts[:, 2] -= z0
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


def furniture(V, space, landmarks, cell=0.05):
    """What stands in the duck's way inside one space: a median top-height map below 1.2 m (upper cabinets don't count),
    cut where the height jumps (seats ~0.45, tables ~0.75, counters ~0.95), each layer as its largest rectangles."""
    (x0, y0), (x1, y1) = np.array(space["polygon"]).min(0), np.array(space["polygon"]).max(0)
    inner = np.zeros_like(space["mask"])  # the outline (pushed to the walls), 5 cm clear of them: wall counters stay
    cv.fillPoly(inner, [np.round((np.array(space["polygon"]) - space["lo"]) / space["cell"]).astype(np.int32)], 1)
    inner = cv.erode(inner, np.ones((3, 3), np.uint8))
    ij = ((V[:, :2] - space["lo"]) / space["cell"]).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < inner.shape[1]) & (ij[:, 1] < inner.shape[0])
    sel = np.zeros(len(V), bool)
    sel[ok] = inner[ij[ok, 1], ij[ok, 0]] > 0
    low = V[sel & (V[:, 2] > 0.15) & (V[:, 2] < 1.2)]
    tall = V[sel & (V[:, 2] > 0.15) & (V[:, 2] < 2.0)]
    rows, cols = int((y1 - y0) / cell) + 1, int((x1 - x0) / cell) + 1
    key = ((low[:, 1] - y0) / cell).astype(int) * cols + ((low[:, 0] - x0) / cell).astype(int)
    order = np.lexsort((low[:, 2], key))
    cells, first, count = np.unique(key[order], return_index=True, return_counts=True)
    hmap = np.zeros((rows, cols), np.float32)
    hmap.flat[cells[count >= 2]] = low[order, 2][(first + count // 2)[count >= 2]]
    out = []
    for lo, hi, guess in ((0.15, 0.62, "seat"), (0.62, 0.86, "table"), (0.86, 1.21, "counter")):
        mask = ((hmap >= lo) & (hmap < hi)).astype(np.uint8)
        mask = cv.morphologyEx(cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        for r0, c0, r1, c1 in floorplan.rects_of(mask.astype(bool), cell, 0.15, 0.3):
            bx0, bx1, by0, by1 = x0 + c0 * cell, x0 + c1 * cell, y0 + r0 * cell, y0 + r1 * cell
            bx0, by0 = (x0 if bx0 < x0 + 0.25 else bx0), (y0 if by0 < y0 + 0.25 else by0)  # flush to the wall it touches
            bx1, by1 = (x1 if bx1 > x1 - 0.25 else bx1), (y1 if by1 > y1 - 0.25 else by1)
            top = float(np.percentile(hmap[r0:r1, c0:c1][hmap[r0:r1, c0:c1] > 0], 90))
            near = [lm for lm in landmarks if bx0 - 0.3 < lm["center"][0] < bx1 + 0.3 and by0 - 0.3 < lm["center"][1] < by1 + 0.3]
            mid = ((bx0 + bx1) / 2, (by0 + by1) / 2)
            label = min(near, key=lambda lm: np.hypot(lm["center"][0] - mid[0], lm["center"][1] - mid[1]))["label"] if near else guess
            if label == "fridge":  # taller than the 1.2 m map
                foot = tall[(tall[:, 0] > bx0) & (tall[:, 0] < bx1) & (tall[:, 1] > by0) & (tall[:, 1] < by1)]
                top = float(np.percentile(foot[:, 2], 95)) if len(foot) else top
            out.append({"label": label, "kind": "table" if "table" in label else "box", "center": [(bx0 + bx1) / 2, (by0 + by1) / 2],
                        "size": [round(bx1 - bx0, 2), round(by1 - by0, 2), round(top, 2)], "yaw": 0.0})
    return out


def build(src, out=None, detect_fn=upright_detect, every=2, extra_markers=None):
    tm, tic = {}, time.time()
    ex = Export(src)
    frames = [f for f in ex.frames if f["tracking"] == "normal"]
    out = Path(out or ROOT / "out" / "lidar" / ex.session)
    out.mkdir(parents=True, exist_ok=True)
    w, h = frames[0]["depth_wh"]
    Kd = kmat(frames[0]["K_depth"])
    vs = cv.VolumeSettings(cv.VolumeType_HashTSDF)
    vs.setIntegrateWidth(w), vs.setIntegrateHeight(h), vs.setRaycastWidth(w), vs.setRaycastHeight(h)
    vs.setCameraIntegrateIntrinsics(Kd), vs.setCameraRaycastIntrinsics(Kd)
    vs.setVoxelSize(VOX), vs.setTsdfTruncateDistance(4 * VOX), vs.setDepthFactor(1.0), vs.setMaxDepth(MAX_D)
    vol = cv.Volume(cv.VolumeType_HashTSDF, vs)
    normals = cv.RgbdNormals_create(h, w, cv.CV_32F, Kd, 5, method=cv.RGBD_NORMALS_RGBD_NORMALS_METHOD_FALS)
    names = sorted({f["room"] for f in frames}, key=[f["room"] for f in frames].index)
    pts, nrm, col, lab, poses = [], [], [], [], []
    for f in frames:
        d, P = ex.load(f)
        vol.integrate(d, P.astype(np.float32))
        P3 = cv.depthTo3d(d, Kd)
        N = normals.apply(P3)[::2, ::2, :3].reshape(-1, 3)
        q = P3[::2, ::2, :3].reshape(-1, 3)
        ok = (d[::2, ::2].ravel() > 0) & np.isfinite(N).all(1)
        q, N = q[ok], N[ok]
        N *= -np.sign((N * q).sum(1))[:, None]  # face the camera
        pts.append(q @ P[:3, :3].T + P[:3, 3])
        nrm.append(N @ P[:3, :3].T)
        col.append(cv.resize(ex.rgb(f), (w, h), interpolation=cv.INTER_AREA)[::2, ::2].reshape(-1, 3)[ok])
        lab.append(np.full(len(q), names.index(f["room"]), np.int8))
        poses.append(P)
    tm["fuse_s"] = time.time() - tic
    # voxel grid: mean position / normal / colour and the majority room label of the frames that saw it
    uk, inv = np.unique(vkey(np.concatenate(pts)), return_inverse=True)
    cnt, (V, Nv, C) = means(inv, len(uk), np.concatenate(pts), np.concatenate(nrm), np.concatenate(col).astype(float))
    votes = np.bincount(inv * len(names) + np.concatenate(lab), minlength=len(uk) * len(names)).reshape(-1, len(names))
    keep = cnt >= 2
    V, Nv, C, L = V[keep], Nv[keep], C[keep], votes[keep].argmax(1)
    Nv /= np.linalg.norm(Nv, axis=1, keepdims=True) + 1e-9
    del pts, nrm, col, lab
    # floor, Manhattan axes, rooms
    z0 = lowest_layer(V[Nv[:, 2] > 0.9, 2])
    V[:, 2] -= z0
    wall = (np.abs(Nv[:, 2]) < 0.3) & (V[:, 2] > 0.3) & (V[:, 2] < 2.0)
    theta = manhattan(Nv[wall])
    R2 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    V[:, :2], Nv[:, :2] = V[:, :2] @ R2.T, Nv[:, :2] @ R2.T
    traj = np.array([P[:3, 3] for P in poses])
    traj[:, :2] = traj[:, :2] @ R2.T
    traj[:, 2] -= z0
    tlab = [f["room"] for f in frames]
    spaces, zones, walls_full, doors = floorplan.plan(V, Nv, traj, tlab)
    walls = [w_[:5] for w_ in walls_full]
    tm["rooms_s"] = time.time() - tic - sum(tm.values())
    # landmarks: detector on keyframes → main surface in 3D → merged by class when they touch → oriented boxes
    found = []
    for f, P in list(zip(frames, poses))[::every]:
        d, _ = ex.load(f, conf=1)  # grazing table tops are only medium confidence
        up = P[:3, :3].T @ [0, 0, 1]
        dets = detect_fn(ex.rgb(f), up, Path(f["rgb"]).stem)
        if dets:
            P3 = cv.depthTo3d(d, Kd)
            for cls, sc, box in dets:
                p = lift(P3, d, box, w / f["rgb_wh"][0], P, z0, R2, walls)
                if p is not None:
                    found.append((cls, sc, p))
    groups = []  # same class + touching (shared 15 cm cells) = one object seen from another side
    for cls, sc, p in sorted(found, key=lambda x: -x[1]):
        k = np.unique(vkey(p, 0.15))
        g = next((g for g in groups if g["cls"] == cls and np.intersect1d(g["keys"], k).size), None)
        if g:
            g["pts"].append(p)
            g["keys"] = np.union1d(g["keys"], k)
        else:
            groups.append({"cls": cls, "keys": k, "pts": [p]})
    landmarks = []
    for g in groups:
        if len(g["pts"]) >= 2:
            center, size, yaw = box3d(np.concatenate(g["pts"]))
            label = name_for(g["cls"], zone_at(zones, center))
            if label and max(size[:2]) >= 0.2:
                landmarks.append({"label": label, "center": center, "size": size, "yaw": yaw, "views": len(g["pts"])})
    # furniture: what stands in the duck's way, from the points themselves (height layers → rectangles)
    objects = []
    for s in spaces:
        objects += furniture(V, s, landmarks)
    tm["objects_s"] = time.time() - tic - sum(tm.values())
    markers = []
    try:
        from twin.markers_from_scan import find_markers
        for m in find_markers(ex, frames, poses):
            c, n = np.array(m["center"], float), np.array(m["normal"], float)
            c[2] -= z0
            c[:2], n[:2] = c[:2] @ R2.T, n[:2] @ R2.T
            markers.append({"id": m["id"], "x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "yaw": float(np.arctan2(n[1], n[0])),
                            "size_m": m["size_m"], "views": m["views"], "room": zone_at(zones, c[:2])})
    except ImportError:
        pass
    # origin = the lower-left corner of the largest space
    o = np.array(max(spaces, key=lambda s: s["mask"].sum())["polygon"]).min(0)
    if extra_markers and not markers:  # placed by hand in the house frame (origin already applied), snapped onto the
        markers = []                    # face of the wall behind them if one is within 30 cm
        for m in json.loads(Path(extra_markers).read_text())["markers"]:
            x, y = m["x"] + o[0], m["y"] + o[1]
            n = np.array([np.cos(m["yaw"]), np.sin(m["yaw"])])
            ax = 0 if abs(n[0]) > abs(n[1]) else 1  # facing ±x: on a wall along y (axis 0)
            along, across = (y, x) if ax == 0 else (x, y)
            faces = [w_[1] + (w_[4] / 2 if w_[5] else 0) * np.sign(n[ax]) for w_ in walls_full  # outer walls: line = face
                     if w_[0] == ax and w_[2] - 0.1 <= along <= w_[3] + 0.1]
            for ob in objects:  # furniture fronts too (axis-aligned boxes)
                c, s = ob["center"], ob["size"]
                if abs(along - c[1 - ax]) <= s[1 - ax] / 2:
                    faces.append(c[ax] + s[ax] / 2 * np.sign(n[ax]))
            near = [f for f in faces if abs(f - across) <= 0.3]
            if near:  # the outermost surface toward the viewer: a marker is never buried in a counter or a wall
                face = max(near, key=lambda f: f * np.sign(n[ax]))
                x, y = (face, y) if ax == 0 else (x, face)
            markers.append({**m, "x": float(x), "y": float(y)})
    r3 = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    shift = lambda p: r3(np.subtract(p, o))  # noqa: E731
    ceiling = float(np.percentile(V[:, 2], 99.5))
    house = {
        "version": 2, "units": "m", "source": f"lidar/{ex.session}",
        "rooms": [{**z, "polygon": [shift(p) for p in z["polygon"]], "rects": [shift(r[:2]) + shift(r[2:]) for r in z["rects"]]}
                  for z in zones],
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
    (out / "house.json").write_text(json.dumps(house, indent=1))
    V[:, :2] -= o
    traj[:, :2] -= o
    top_view(out / "top.png", house, V[wall], traj, [names.index(f["room"]) for f in frames])
    # fused TSDF surface, raycast from every 8th pose, 3 cm → cloud.ply, coloured from the voxel grid (5 cm lookup)
    tp, tn = [], []
    for P in poses[::8]:
        q, n = (a.reshape(-1, 4)[:, :3].astype(float) for a in vol.raycast(P.astype(np.float32)))
        ok = np.isfinite(q).all(1) & (q[:, 2] > 0)
        tp.append(q[ok] @ P[:3, :3].T + P[:3, 3])
        tn.append(n[ok] @ P[:3, :3].T)
    tp, tn = np.concatenate(tp), np.concatenate(tn)
    tk, tinv = np.unique(vkey(tp, 0.03), return_inverse=True)
    _, (tp, tn) = means(tinv, len(tk), tp, tn)
    tp[:, 2] -= z0
    tp[:, :2], tn[:, :2] = tp[:, :2] @ R2.T - o, tn[:, :2] @ R2.T
    ck, cinv = np.unique(vkey(V, 0.05), return_inverse=True)
    _, (cc,) = means(cinv, len(ck), C)
    j = np.clip(np.searchsorted(ck, vkey(tp, 0.05)), 0, len(ck) - 1)
    rgb = np.where((ck[j] == vkey(tp, 0.05))[:, None], cc[j][:, ::-1] / 255, 0.6)
    cv.savePointCloud(str(out / "cloud.ply"), *(a.astype(np.float32).reshape(-1, 1, 3) for a in (tp, tn, rgb)))
    tm["export_s"] = time.time() - tic - sum(tm.values())
    report = {
        "session": ex.session, "frames": len(ex.frames), "frames_used": len(frames), "voxels": int(len(V)),
        "tsdf_points": int(len(tp)), "floor_offset_m": round(z0, 3), "manhattan_deg": round(np.degrees(theta), 2),
        "spaces": {s["name"]: {"w_m": round(float(np.ptp(np.array(s["polygon"])[:, 0])), 3), "l_m": round(float(np.ptp(np.array(s["polygon"])[:, 1])), 3),
                               "area_m2": round(float(s["mask"].sum()) * floorplan.CELL ** 2, 2), "corners": len(s["polygon"])} for s in spaces},
        "zones": {z["name"]: z["space"] for z in zones}, "doors": len(doors), "objects": len(objects),
        "landmarks": len(landmarks), "markers": len(markers), "walls": len(walls),
        "timings_s": {k: round(v, 2) for k, v in tm.items()}, "cv_version": cv.__version__}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return house, out


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
    for i in range(1, len(traj)):
        cv.line(img, to(traj[i - 1]), to(traj[i]), [c * 0.6 for c in colors[tlab[i] % len(colors)]], 2)
    cv.circle(img, to(traj[0]), 6, (200, 80, 30), -1)
    for r in house["rooms"]:
        p = np.array(r["polygon"])
        w, l = np.ptp(p, 0)
        cv.putText(img, f"{r['name']} {w:.2f}x{l:.2f}m", to(p.mean(0) - [0.8, 0]), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 0), 2)
    cv.circle(img, to((0, 0)), 5, (0, 0, 255), -1)
    cv.imwrite(str(path), img)


if __name__ == "__main__":
    house, out = build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(json.loads((out / "report.json").read_text()), indent=1))
    print("wrote", out)
