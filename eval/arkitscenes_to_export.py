"""ARKitScenes (Apple, iPad LiDAR, github.com/apple/ARKitScenes, non-commercial licence) → our iPhone export format.

python eval/arkitscenes_to_export.py fetch <video_id> <scene_dir> [Validation|Training] [max_frames]
    only the frames we use: lowres_wide.traj, the annotation, and ~2.5 frames/s of lowres_depth + confidence + vga_wide
    (640×480) with their intrinsics, read out of the raw zips by HTTP range requests (no full download).
python eval/arkitscenes_to_export.py <scene_dir> <out.zip>
    → out.zip (lidar/README.md format, one zone "Home") + <out>.labels.json (annotated boxes, same ARKit world as T).
python eval/arkitscenes_to_export.py match <labels.json> <house_dir>
    candidates.json / house.json landmarks of a build → nearest annotated box, by class.

Conventions (checked on the data, see check_up()): a .traj line is `t, axis-angle, translation` of world→camera for an
OpenCV camera (x right, y down, z forward); the scene world is gravity-aligned with z up. Our export wants ARKit's
camera→world with y up and an ARKit camera (x right, y up, z back): T = A · inv(E) · diag(1, -1, -1, 1), A: z-up → y-up.
"""
import json
import shutil
import sys
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # scene world (z up) → ARKit world (y up): (x, z, -y)
DET_TO_ARK = {"couch": {"sofa"}, "refrigerator": {"refrigerator"}, "tv": {"tv_monitor"}, "bed": {"bed"}, "toilet": {"toilet"},
              "sink": {"sink"}, "oven": {"oven", "stove"}, "microwave": {"oven"}, "dining table": {"table"},
              "chair": {"chair", "stool"}, "bench": {"chair", "stool"}}


class RangeFile:
    """A remote file that zipfile can seek in: every read is one HTTP range request."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        self.pos += len(data)
        return data


def stamps(names):
    return {round(float(Path(n).stem.split("_")[1]), 3): n for n in names if n.endswith((".png", ".pincam"))}


def fetch(vid, dst, split="Validation", max_frames=300):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    base = f"{URL}/{split}/{vid}"
    for name, out in ((f"{vid}_3dod_annotation.json", "annotation.json"), ("lowres_wide.traj", "lowres_wide.traj")):
        (dst / out).write_bytes(urllib.request.urlopen(f"{base}/{name}", timeout=120).read())
    traj = [float(s.split()[0]) for s in (dst / "lowres_wide.traj").read_text().splitlines() if s.strip()]
    zips = {a: zipfile.ZipFile(RangeFile(f"{base}/{a}.zip")) for a in
            ("lowres_depth", "confidence", "lowres_wide_intrinsics", "vga_wide", "vga_wide_intrinsics")}
    idx = {a: stamps(z.namelist()) for a, z in zips.items()}
    vga_t = np.array(sorted(idx["vga_wide"]))
    step = max(0.4, (traj[-1] - traj[0]) / max_frames)
    want, last = [], -1e9
    for t in traj:
        k = round(t, 3)
        d = next((k + e for e in (0, -0.001, 0.001) if round(k + e, 3) in idx["lowres_depth"]), None)
        if t - last < step or d is None or round(d, 3) not in idx["confidence"] or round(d, 3) not in idx["lowres_wide_intrinsics"]:
            continue
        v = vga_t[np.argmin(np.abs(vga_t - t))]
        if abs(v - t) > 0.02 or v not in idx["vga_wide_intrinsics"]:
            continue
        want.append({"t": t, "d": round(d, 3), "v": float(v)})
        last = t
    jobs = [(a, f[key]) for f in want for a, key in (("lowres_depth", "d"), ("confidence", "d"), ("lowres_wide_intrinsics", "d"),
                                                      ("vga_wide", "v"), ("vga_wide_intrinsics", "v"))]
    def get(job):  # one range request per member: local header + data, from the central directory's offsets
        a, key = job
        info = zips[a].getinfo(idx[a][key])
        n = 30 + len(info.filename.encode()) + 512 + info.compress_size
        req = urllib.request.Request(f"{base}/{a}.zip", headers={"Range": f"bytes={info.header_offset}-{info.header_offset + n - 1}"})
        raw = urllib.request.urlopen(req, timeout=120).read()
        start = 30 + int.from_bytes(raw[26:28], "little") + int.from_bytes(raw[28:30], "little")
        data = raw[start:start + info.compress_size]
        data = zlib.decompress(data, -15) if info.compress_type == zipfile.ZIP_DEFLATED else data
        (dst / info.filename).parent.mkdir(parents=True, exist_ok=True)
        (dst / info.filename).write_bytes(data)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, jobs))
    (dst / "frames.json").write_text(json.dumps({"video_id": vid, "split": split, "frames": want}, indent=0))
    print(f"{vid}: {len(want)} frames, {sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()) / 1e6:.0f} MB in {dst}")


def poses(scene):
    """timestamp → OpenCV camera→scene world (4×4)."""
    out = {}
    for s in (scene / "lowres_wide.traj").read_text().splitlines():
        v = [float(x) for x in s.split()]
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = cv.Rodrigues(np.array(v[1:4]))[0], v[4:7]
        out[v[0]] = np.linalg.inv(E)
    return out


def pincam(path):
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    return int(w), int(h), [fx, fy, cx, cy]


def check_up(scene, meta, P):
    """Lowest points per scene axis (z ≈ the floor, about the annotated boxes' bottoms, when z is up)."""
    pts = []
    for f in meta["frames"][::max(1, len(meta["frames"]) // 20)]:
        vid = meta["video_id"]
        d = cv.imread(str(scene / f"lowres_depth/{vid}_{f['d']:.3f}.png"), cv.IMREAD_UNCHANGED) / 1000.0
        _, _, k = pincam(scene / f"lowres_wide_intrinsics/{vid}_{f['d']:.3f}.pincam")
        q = cv.depthTo3d(d.astype(np.float32), np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], np.float32)).reshape(-1, 3)
        q = q[(q[:, 2] > 0.2) & (q[:, 2] < 5)]
        T = P[f["t"]]
        pts.append(q @ T[:3, :3].T + T[:3, 3])
    pts = np.concatenate(pts)
    return {ax: round(float(np.percentile(pts[:, i], 1)), 2) for i, ax in enumerate("xyz")}  # 1st percentile per axis


def convert(scene, out):
    scene, out = Path(scene), Path(out)
    meta = json.loads((scene / "frames.json").read_text())
    vid, P = meta["video_id"], poses(scene)
    session = out.stem
    root = out.parent / f"_tmp_{session}" / session
    shutil.rmtree(root.parent, ignore_errors=True)
    for sub in ("rgb", "depth", "conf"):
        (root / sub).mkdir(parents=True)
    lines = []
    for i, f in enumerate(meta["frames"]):
        name = f"{i:06d}"
        img = cv.imread(str(scene / f"vga_wide/{vid}_{f['v']:.3f}.png"))
        cv.imwrite(str(root / f"rgb/{name}.jpg"), img, [cv.IMWRITE_JPEG_QUALITY, 85])
        rw, rh, K = pincam(scene / f"vga_wide_intrinsics/{vid}_{f['v']:.3f}.pincam")
        dw, dh, Kd = pincam(scene / f"lowres_wide_intrinsics/{vid}_{f['d']:.3f}.pincam")
        d = cv.imread(str(scene / f"lowres_depth/{vid}_{f['d']:.3f}.png"), cv.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        c = cv.imread(str(scene / f"confidence/{vid}_{f['d']:.3f}.png"), cv.IMREAD_UNCHANGED).astype(np.uint8)
        assert d.shape == (dh, dw) == c.shape, (d.shape, c.shape, dw, dh)
        (root / f"depth/{name}.bin").write_bytes(d.astype(np.float16).tobytes())
        (root / f"conf/{name}.bin").write_bytes(c.tobytes())  # ARKit's own confidence (0 low, 1 medium, 2 high)
        T = np.eye(4)
        Tc = P[f["t"]]
        T[:3, :3], T[:3, 3] = A @ Tc[:3, :3] @ np.diag([1, -1, -1]), A @ Tc[:3, 3]
        lines.append({"i": i, "t": round(f["t"] - meta["frames"][0]["t"], 3), "rgb": f"rgb/{name}.jpg", "depth": f"depth/{name}.bin",
                      "conf": f"conf/{name}.bin", "rgb_wh": [rw, rh], "K": K, "depth_wh": [dw, dh], "K_depth": Kd,
                      "T": T.ravel().tolist(), "tracking": "normal", "room": "Home", "h": None})
    (root / "frames.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (root / "events.jsonl").write_text(json.dumps({"t": 0, "i": 0, "event": "room", "room": "Home", "T": lines[0]["T"]}) + "\n")
    (root / "session.json").write_text(json.dumps({"session": session, "device": f"ARKitScenes {vid} (iPad Pro)", "fps": 2.5, "format": 2}))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            z.write(p, p.relative_to(root.parent))
    shutil.rmtree(root.parent)
    boxes = []
    for b in json.loads((scene / "annotation.json").read_text())["data"]:
        s = b["segments"]["obbAligned"]
        R = np.array(s["normalizedAxes"]).reshape(3, 3)
        boxes.append({"label": b["label"], "uid": b.get("uid"), "center": (A @ np.array(s["centroid"])).round(4).tolist(),
                      "size": np.round(s["axesLengths"], 4).tolist(), "rotation": (A @ R.T).round(5).tolist()})
    labels = out.with_suffix(".labels.json")
    labels.write_text(json.dumps({"video_id": vid, "frame": "ARKit world (y up, metres), the same as the export's T. rotation: "
                                  "3×3 whose columns are the box axes, size along those axes (full lengths).",
                                  "up_check": check_up(scene, meta, P), "boxes": boxes}, indent=1))
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB, {len(lines)} frames), {labels} ({len(boxes)} boxes)")


def match(labels, house_dir):
    """Annotated boxes → the house frame (report.json scan_frame), then each candidate → the box it sits in (+30 cm)."""
    house_dir = Path(house_dir)
    lab = json.loads(Path(labels).read_text())
    rep = json.loads((house_dir / "report.json").read_text())
    th, o = rep["scan_frame"]["theta"], np.array(rep["scan_frame"]["origin"])
    R2 = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # ARKit y up → house z up (twin/scan_io.py)
    boxes = []
    for b in lab["boxes"]:
        R = M @ np.array(b["rotation"])
        corners = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * b["size"] @ R.T + M @ b["center"]
        xy = corners[:, :2] @ R2.T - o
        boxes.append({**b, "lo": xy.min(0), "hi": xy.max(0)})
    cands = json.loads((house_dir / "candidates.json").read_text())
    rows, per = [], {}
    for c in cands:
        p = np.array(c["center"])
        inside = [b for b in boxes if (p >= b["lo"] - 0.3).all() and (p <= b["hi"] + 0.3).all()]
        best = min(inside, key=lambda b: np.linalg.norm(p - (b["lo"] + b["hi"]) / 2), default=None)
        same = [b for b in inside if b["label"] in DET_TO_ARK.get(c["cls"], set())]
        verdict = "right class" if same else ("other box: " + best["label"] if best else "no box")
        rows.append({"id": c["id"], "cls": c["cls"], "label": c["label"], "views": c["views"], "kept": c["decision"]["keep"],
                     "match": verdict})
        k = per.setdefault(c["cls"], {"candidates": 0, "right_class": 0, "other_box": 0, "no_box": 0})
        k["candidates"] += 1
        k["right_class" if same else ("other_box" if best else "no_box")] += 1
    recall = {}
    for b in boxes:
        hit = any(b["label"] in DET_TO_ARK.get(c["cls"], set()) and (np.array(c["center"]) >= b["lo"] - 0.3).all()
                  and (np.array(c["center"]) <= b["hi"] + 0.3).all() for c in cands)
        r = recall.setdefault(b["label"], [0, 0])
        r[0] += hit
        r[1] += 1
    res = {"candidates": len(cands), "by_detector_class": per,
           "annotated_boxes_found": {k: f"{a}/{n}" for k, (a, n) in sorted(recall.items())}, "rows": rows}
    (house_dir / "label_match.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or ["Validation"]), *[int(x) for x in sys.argv[5:6]])
    elif sys.argv[1] == "match":
        r = match(sys.argv[2], sys.argv[3])
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    else:
        convert(sys.argv[1], sys.argv[2])
