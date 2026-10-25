"""Read one iPhone LiDAR export (app, format in lidar/README.md): a zip or an unpacked folder.

ARKit world is y up (gravity), camera x right / y up / z back. OpenCV camera is x right / y down / z forward,
so T_cv = T_ar · diag(1,-1,-1,1); the house frame is z up: (x, y, z) = (x_ar, -z_ar, y_ar).
"""
import json
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np

MAX_D = 5.0
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # ARKit world (y up) → house (z up)
FLIP = np.diag([1.0, -1, -1])  # ARKit camera → OpenCV camera


def kmat(k):
    return np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], np.float32)


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
        self.labels = self.json("labels.json") or {}
        self.extra_tags = []  # tags added later on the results page (same format as the app's)
        fix = {i: r["room"] for r in self.labels.get("ranges", []) for i in range(r["from"], r["to"] + 1)}
        for f in self.frames:  # zones fixed in the app's review screen win over the live taps
            f["room"] = fix.get(f["i"], f["room"])

    def read(self, name):
        return self.z.read(self.pre + name) if self.z else Path(self.pre + name).read_bytes()

    def json(self, name):
        try:
            return json.loads(self.read(name))
        except (KeyError, FileNotFoundError):
            return None

    def events(self):
        try:
            return [json.loads(s) for s in self.read("events.jsonl").decode().splitlines() if s.strip()]
        except (KeyError, FileNotFoundError):
            return []

    def tags(self):
        """Objects the user labelled under the crosshair: [{i, label, uv, room}]. The review screen's list wins."""
        app = self.labels["tags"] if "tags" in self.labels else [e for e in self.events() if e.get("event") == "tag"]
        return clean_tags((app if isinstance(app, list) else []) + self.extra_tags)

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


def clean_tags(raw):
    """Drop malformed tags (they come from a phone file); trim labels to 40 characters (they reach a model's prompt)."""
    out = []
    for t in raw if isinstance(raw, list) else []:
        if not isinstance(t, dict) or type(t.get("i")) is not int or not isinstance(t.get("label"), str) or not t["label"].strip():
            continue
        uv = t.get("uv") or [0.5, 0.5]
        if not (isinstance(uv, list) and len(uv) == 2 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in uv)):
            continue
        out.append({"i": t["i"], "label": t["label"].strip()[:40], "uv": uv, "room": t.get("room"), "web": bool(t.get("web"))})
    return out
