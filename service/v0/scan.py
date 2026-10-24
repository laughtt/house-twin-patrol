"""v0 markerless scan (OpenCV 5): spin video → lens + rotations → depth → floor, walls, object heights.

Sizes come out in "model metres" (the depth model's guess). agent.py turns them into real metres.
"""
from pathlib import Path

import cv2 as cv
import numpy as np

MODELS = Path(__file__).parent.parent / "models"
DEPTH_WH = (504, 280)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
FURNITURE = {13: "bench", 56: "chair", 57: "couch", 59: "bed", 60: "dining table", 61: "toilet",
             69: "oven", 71: "sink", 72: "refrigerator"}
LANDMARK = {**FURNITURE, 58: "potted plant", 62: "tv", 68: "microwave", 74: "clock"}  # things a robot can find itself by
# the phone circles ~10-15 cm in front of the turning body, so every wall reads that much too close
PHONE_SWING_M = 0.12
aruco = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), cv.aruco.DetectorParameters())
nets = {}

GRID, STRIDE = [], []
for _s in (8, 16, 32):
    _yv, _xv = np.mgrid[:640 // _s, :640 // _s]
    GRID.append(np.stack([_xv, _yv], -1).reshape(-1, 2))
    STRIDE.append(np.full((_xv.size, 1), _s))
GRID, STRIDE = np.concatenate(GRID), np.concatenate(STRIDE)


def net(name):
    if name not in nets:
        nets[name] = cv.dnn.readNet(str(MODELS / name))
    return nets[name]


def hide_markers(img):
    # v0 must not see the markers (or the paper they're printed on)
    corners, _, _ = aruco.detectMarkers(img)
    for c in corners:
        m = c[0].mean(0)
        cv.fillConvexPoly(img, ((c[0] - m) * 1.6 + m).astype(np.int32), (128, 128, 128))
    return img


def read_frames(video, seconds=24, every_s=0.4, width=960):  # only the spin at the start of the clip
    cap = cv.VideoCapture(str(video))
    if cap.get(cv.CAP_PROP_FRAME_HEIGHT) > cap.get(cv.CAP_PROP_FRAME_WIDTH):
        # iPhone Safari tags sideways-held clips as rotated; the raw pixels are the upright landscape picture
        cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    step = max(1, round(fps * every_s))
    frames = []
    for i in range(int(fps * seconds)):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if f.shape[0] > f.shape[1]:
                raise ValueError("film in landscape")
            frames.append(hide_markers(cv.resize(f, (width, round(f.shape[0] * width / f.shape[1])),
                                                 interpolation=cv.INTER_AREA)))
    sharp = [cv.Laplacian(cv.cvtColor(f, cv.COLOR_BGR2GRAY), cv.CV_64F).var() for f in frames]
    return [f for f, s in zip(frames, sharp) if s > 0.3 * np.median(sharp)]


def depth(img):
    x = cv.resize(img, DEPTH_WH, interpolation=cv.INTER_CUBIC)[:, :, ::-1] / 255.0
    net("depth_metric_indoor_s.onnx").setInput(((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))
    return net("depth_metric_indoor_s.onnx").forward()[0]


def depth_pro(img):
    """Apple Depth Pro (research licence): metric depth + the lens from one frame. ~50 s and 5 GB on 4 cores."""
    import onnxruntime as ort
    if "pro" not in nets:
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = so.enable_mem_pattern = False  # with pooling it needs > 7.5 GB
        nets["pro"] = ort.InferenceSession(str(MODELS / "depthpro_int8.onnx"), so, providers=["CPUExecutionProvider"])
    x = cv.resize(img, (1536, 1536), interpolation=cv.INTER_LINEAR)[:, :, ::-1] / 255.0
    d, fpx = nets["pro"].run(None, {"pixel_values": ((x - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)})
    return cv.resize(d[0], DEPTH_WH), float(np.degrees(2 * np.arctan(768 / fpx.ravel()[0])))


def detect(img, conf=0.45, classes=FURNITURE):
    r = min(640 / img.shape[0], 640 / img.shape[1])
    h, w = round(img.shape[0] * r), round(img.shape[1] * r)
    pad = np.full((640, 640, 3), 114, np.uint8)
    pad[:h, :w] = cv.resize(img, (w, h))
    net("yolox_s.onnx").setInput(pad.transpose(2, 0, 1)[None].astype(np.float32))
    d = net("yolox_s.onnx").forward()[0]
    xy, wh = (d[:, :2] + GRID) * STRIDE / r, np.exp(d[:, 2:4]) * STRIDE / r
    scores = d[:, 4:5] * d[:, 5:]
    cls, sc = scores.argmax(1), scores.max(1)
    ok = np.nonzero((sc > conf) & np.isin(cls, list(classes)))[0]
    boxes = np.c_[xy[ok] - wh[ok] / 2, wh[ok]]
    keep = cv.dnn.NMSBoxesBatched(boxes.tolist(), sc[ok].tolist(), cls[ok].tolist(), conf, 0.45)
    return [(classes[int(cls[ok[i]])], float(sc[ok[i]]),
             [*boxes[i, :2], *(boxes[i, :2] + boxes[i, 2:])]) for i in np.ravel(keep)]


def register(frames):
    """Pure-rotation panorama solve: each frame's rotation (camera ray → world) + one focal length."""
    n = len(frames)
    feats = [cv.detail.computeImageFeatures2(cv.ORB_create(1500), f) for f in frames]
    mask = np.zeros((n, n), np.uint8)
    for i in range(n):
        mask[i, i + 1:i + 5] = 1
    mask[:15, -15:] = 1  # loop closure: the end of the circle sees the start again
    pairs = cv.detail_BestOf2NearestMatcher(False, 0.3).apply2(feats, np.triu(mask, 1))
    keep = list(np.ravel(cv.detail.leaveBiggestComponent(feats, pairs, 1.0)))
    if len(keep) < n:
        sub, f, Rs, feats, pairs = register([frames[k] for k in keep])
        return [keep[i] for i in sub], f, Rs, feats, pairs
    _, cams = cv.detail_HomographyBasedEstimator().apply(feats, pairs, None)
    for c in cams:
        c.R = c.R.astype(np.float32)
    ba = cv.detail_BundleAdjusterRay()
    ba.setConfThresh(1.0)
    _, cams = ba.apply(feats, pairs, cams)
    Rs = cv.detail.waveCorrect([c.R for c in cams], cv.detail.WAVE_CORRECT_HORIZ)
    return keep, float(np.median([c.focal for c in cams])), [np.asarray(R, np.float64) for R in Rs], feats, pairs


class Scan:
    def __init__(self, frames, depth_fn=depth):
        keep, self.f, self.R, feats, pairs = register(frames)
        self.frames = [frames[k] for k in keep]
        h, w = self.frames[0].shape[:2]
        self.K = np.array([[self.f, 0, w / 2], [0, self.f, h / 2], [0, 0, 1]])
        self.Kd = np.diag([DEPTH_WH[0] / w, DEPTH_WH[1] / h, 1]) @ self.K
        self.hfov = float(np.degrees(2 * np.arctan(w / 2 / self.f)))
        self.depth = [depth_fn(f) for f in self.frames]
        self.s = self._align_scales(feats, pairs)
        self.pro = self._depth_pro() if (MODELS / "depthpro_int8.onnx").exists() else None
        fwd = np.array([R[:, 2] for R in self.R])
        yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
        self.spin_deg = float(np.degrees(yaw.max() - yaw.min()))
        P = np.concatenate([self.points(k) for k in range(len(self.frames))])
        P = P[np.linalg.norm(P, axis=1) < 10]
        self.n, self.cam_h = self._floor(P, -np.mean([R[:, 1] for R in self.R], 0))
        a = fwd[0] - fwd[0] @ self.n * self.n
        self.e1 = a / np.linalg.norm(a)
        self.e2 = np.cross(self.n, self.e1)
        hgt = P @ self.n + self.cam_h
        self.Q = (P[(hgt > 0.3) & (hgt < 2.0)] @ np.c_[self.e1, self.e2])
        self.theta = self._wall_angle(self.Q)
        self.walls = self._walls(self.Q, self.theta)
        self.size = (self.walls["+a"] + self.walls["-a"], self.walls["+b"] + self.walls["-b"])

    def _depth_pro(self, n=2):
        # Depth Pro on n frames spread round the circle: how far off the small model's metres are, and its lens
        ratios, fovs = [], []
        for k in np.linspace(0, len(self.frames) - 1, n + 1)[:-1].astype(int):
            d, fov = depth_pro(self.frames[k])
            ok = (d > 0.1) & (d < 20)
            fovs.append(fov)
            if ok.mean() > 0.2:  # a frame it can't read (all too near or too far) says nothing about size
                ratios.append(float(np.median(d[ok] / (self.s[k] * self.depth[k][ok]))))
        nets.pop("pro")  # give the memory back
        if not ratios:
            return None
        return {"scale": float(np.median(ratios)), "spread": float(np.std(np.log(ratios))),
                "hfov_deg": round(float(np.median(fovs)), 1), "frames": n, "used": len(ratios)}

    def rays(self, us, vs):
        return np.c_[us, vs, np.ones_like(us)] @ np.linalg.inv(self.Kd).T

    def points(self, k, box=None, step=6):
        dh, dw = self.depth[k].shape
        x0, y0, x1, y1 = (0, 0, dw, dh) if box is None else box
        vs, us = np.mgrid[int(y0) + step // 2:int(y1):step, int(x0) + step // 2:int(x1):step]
        us, vs = us.ravel().astype(float), vs.ravel().astype(float)
        z = self.depth[k][vs.astype(int), us.astype(int)]
        return self.s[k] * (self.rays(us, vs) * z[:, None]) @ self.R[k].T

    def _align_scales(self, feats, pairs):
        # each frame's depth may be off by its own factor; overlaps tie them together, mean anchored at 1
        n = len(self.frames)
        rows, rhs = [], []
        sx, sy = self.Kd[0, 0] / self.K[0, 0], self.Kd[1, 1] / self.K[1, 1]
        for p in pairs:
            i, j = p.src_img_idx, p.dst_img_idx
            if i < 0 or j <= i or p.confidence < 1.0:
                continue
            inl = np.ravel(p.inliers_mask).astype(bool)
            m = [mm for mm, ok in zip(p.matches, inl) if ok]
            if len(m) < 15:
                continue
            pi = np.array([feats[i].keypoints[mm.queryIdx].pt for mm in m]) * [sx, sy]
            pj = np.array([feats[j].keypoints[mm.trainIdx].pt for mm in m]) * [sx, sy]
            ri = self._range(i, pi)
            rj = self._range(j, pj)
            row = np.zeros(n)
            row[i], row[j] = 1, -1
            rows.append(row)
            rhs.append(np.median(np.log(rj / ri)))
        rows.append(np.full(n, 10.0))
        rhs.append(0.0)
        return np.exp(np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0])

    def _range(self, k, uv):
        dh, dw = self.depth[k].shape
        u = np.clip(uv[:, 0], 0, dw - 1)
        v = np.clip(uv[:, 1], 0, dh - 1)
        z = self.depth[k][v.astype(int), u.astype(int)]
        return z * np.linalg.norm(self.rays(u, v), axis=1)

    @staticmethod
    def _floor(P, up):
        # the lowest big horizontal layer, not the biggest one (a near table top has more pixels than the floor)
        up = up / np.linalg.norm(up)
        below = -(P @ up)
        hist, edges = np.histogram(below, bins=np.arange(0.3, 3.0, 0.02))
        hist = np.convolve(hist, np.ones(3), "same")
        h0 = edges[np.nonzero(hist >= 0.15 * hist.max())[0][-1]] + 0.01
        q = P[np.abs(below - h0) < 0.04]
        n = np.linalg.svd(q - q.mean(0))[2][2]
        n = n * np.sign(n @ up)
        return n, float(-n @ q.mean(0))

    @staticmethod
    def _wall_angle(Q, res=0.02):
        Q = Q[np.abs(Q).max(1) < 8]
        idx = ((Q - Q.min(0)) / res).astype(int)
        img = np.zeros(idx.max(0)[::-1] + 1, np.float32)
        np.add.at(img, (idx[:, 1], idx[:, 0]), 1)
        img = cv.GaussianBlur(img, (0, 0), 2)
        gx, gy = cv.Sobel(img, cv.CV_32F, 1, 0), cv.Sobel(img, cv.CV_32F, 0, 1)
        ang = (np.degrees(np.arctan2(gy, gx)) % 90).astype(int).ravel() % 90
        hist = np.bincount(ang, np.hypot(gx, gy).ravel(), 90)
        hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid")
        return float(np.radians(np.argmax(hist)))

    @staticmethod
    def _walls(Q, theta):
        # per direction: the farthest dense band of wall-height points
        c, s = np.cos(theta), np.sin(theta)
        a, b = Q @ [c, s], Q @ [-s, c]
        out = {}
        for name, u, v in [("+a", a, b), ("-a", -a, b), ("+b", b, a), ("-b", -b, a)]:
            sel = (u > 0.3) & (np.abs(v) < u * np.tan(np.radians(40)))
            hist, edges = np.histogram(u[sel], bins=np.arange(0.3, 10, 0.05))
            hist = np.convolve(hist, np.ones(3), "same")
            i = np.nonzero(hist >= 0.3 * hist.max())[0][-1]
            while i > 0 and hist[i - 1] >= hist[i]:
                i -= 1
            band = u[sel][np.abs(u[sel] - edges[i] - 0.025) < 0.1]
            out[name] = float(np.median(band)) + PHONE_SWING_M
        return out

    def measure(self, k, box):
        """Height of an object's top above the floor (model metres) + its floor position, from one box."""
        x0, y0, x1, y1 = box
        h, w = self.frames[k].shape[:2]
        if y0 < 3 or x1 - x0 < 8 or y1 - y0 < 8:
            return None
        sx, sy = self.Kd[0, 0] / self.K[0, 0], self.Kd[1, 1] / self.K[1, 1]
        cx, bw, bh = (x0 + x1) / 2, x1 - x0, y1 - y0
        P = self.points(k, ((cx - 0.3 * bw) * sx, (y0 + 0.05 * bh) * sy, (cx + 0.3 * bw) * sx, y1 * sy), step=2)
        if len(P) < 10:
            return None
        hor = np.linalg.norm(P - np.outer(P @ self.n, self.n), axis=1)
        near = np.percentile(hor, 20)
        D = float(np.median(hor[hor <= near * 1.15]))
        ray = self.s[k] * self.R[k] @ np.linalg.inv(self.K) @ [cx, y0, 1]
        up = ray @ self.n
        if up > 0:  # top edge above eye level (door, fridge): the edge ray at the object's distance
            top = self.cam_h + D * up / np.linalg.norm(ray - up * self.n)
        else:  # below eye level the top surface is visible: highest points of the object itself
            obj = P[(hor > D - 0.1) & (hor < D + 1.2)]
            top = float(np.percentile(obj @ self.n + self.cam_h, 95))
        sel = P[hor <= near * 1.15]
        return float(top), (np.median(sel @ self.e1), np.median(sel @ self.e2)), D

    def objects(self, every=3):
        found = []
        for k in range(0, len(self.frames), every):
            for cls, sc, box in detect(self.frames[k]):
                m = self.measure(k, box)
                if m:
                    found.append({"cls": cls, "score": sc, "k": k, "box": box, "top": m[0], "xy": m[1]})
        groups = []
        for d in sorted(found, key=lambda d: -d["score"]):
            g = next((g for g in groups if g[0]["cls"] == d["cls"]
                      and np.hypot(*np.subtract(g[0]["xy"], d["xy"])) < 0.7), None)
            (g.append(d) if g else groups.append([d]))
        out = []
        for i, g in enumerate(groups):
            best = max(g, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            out.append({"id": f"obj{i + 1}", "cls": g[0]["cls"], "top_model_m": float(np.median([d["top"] for d in g])),
                        "views": len(g), "k": best["k"], "box": [float(v) for v in best["box"]],
                        "xy": [float(v) for v in np.median([d["xy"] for d in g], 0)]})
        return out

    def top_view(self, path, objects=(), px=60):
        c, s = np.cos(self.theta), np.sin(self.theta)
        a, b = self.Q @ [c, s], self.Q @ [-s, c]
        W = self.walls
        lo = np.array([-W["-a"] - 0.5, -W["-b"] - 0.5])
        size = (np.array([W["+a"] + W["-a"], W["+b"] + W["-b"]]) + 1) * px
        img = np.full((int(size[1]), int(size[0]), 3), 255, np.uint8)
        to = lambda x, y: (int((x - lo[0]) * px), int(size[1] - (y - lo[1]) * px))
        for x, y in zip(a[::5], b[::5]):
            p = to(x, y)
            if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                img[p[1], p[0]] = (170, 170, 170)
        cv.rectangle(img, to(-W["-a"], W["+b"]), to(W["+a"], -W["-b"]), (60, 140, 40), 3)
        cv.circle(img, to(0, 0), 7, (200, 80, 30), -1)
        for o in objects:
            x, y = o["xy"][0] * c + o["xy"][1] * s, -o["xy"][0] * s + o["xy"][1] * c
            cv.circle(img, to(x, y), 6, (40, 40, 200), -1)
            cv.putText(img, o["id"], to(x + 0.1, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 200), 1)
        cv.imwrite(str(path), img)
