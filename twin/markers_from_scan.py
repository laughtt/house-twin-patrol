"""ArUco markers (DICT_4X4_50) in a LiDAR scan → id, centre, facing and measured size, in the z-up world of the
build_house.Export.load poses. Corners are lifted onto the wall plane fitted to the LiDAR depth; solvePnP with the
id's printed size when the depth is missing.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python twin/markers_from_scan.py <scan.zip|dir> [truth.json]
"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin.scan_io import Export, kmat  # noqa: E402

_params = cv.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
DET = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), _params)
SQUARE = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32) / 2  # TL TR BR BL, IPPE_SQUARE order


def size_rule(i):
    return 0.10 if 10 <= i <= 29 else 0.20 if 30 <= i <= 39 else 0.15


def corners3d(c, f, d, size):
    """4 RGB corners → 4 camera points: rays onto the plane fitted to the depth under the marker (and its border)."""
    K = kmat(f["K"])
    rays = np.c_[c, np.ones(4)] @ np.linalg.inv(K).T
    q = (rays @ kmat(f["K_depth"]).T)[:, :2]
    mask = np.zeros(d.shape, np.uint8)
    cv.fillConvexPoly(mask, np.round((q - q.mean(0)) * 1.25 + q.mean(0)).astype(np.int32), 1)
    v, u = np.nonzero(mask & (d > 0))
    if len(v) < 12:
        ok, rv, t = cv.solvePnP(SQUARE * size, c, K, None, flags=cv.SOLVEPNP_IPPE_SQUARE)
        return SQUARE * size @ cv.Rodrigues(rv)[0].T + t.ravel(), False
    X = cv.depthTo3d(d, kmat(f["K_depth"]))[v, u, :3]
    for _ in range(3):  # plane fit, dropping the noisy edge pixels
        m = X.mean(0)
        n = np.linalg.svd(X - m)[2][2]
        r = np.abs((X - m) @ n)
        X = X[r <= max(3 * np.median(r), 0.005)]
    return rays * (m @ n / (rays @ n))[:, None], True


def find_markers(ex, frames, poses, sizes=None, every=1, min_px=25):
    views = {}
    for f, P in list(zip(frames, poses))[::every]:
        cs, ids, _ = DET.detectMarkers(ex.rgb(f))
        if ids is None:
            continue
        d = None
        for c, i in zip(cs, ids.ravel().tolist()):
            c = c[0].astype(np.float64)
            if np.linalg.norm(c - np.roll(c, 1, 0), axis=1).mean() < min_px:
                continue  # too far away for good corners
            d = ex.load(f, conf=1)[0] if d is None else d
            X, lidar = corners3d(c, f, d, (sizes or {}).get(i, size_rule(i)))
            W = X @ P[:3, :3].T + P[:3, 3]
            n = np.cross(W[2] - W[0], W[1] - W[3])
            n *= np.sign(n @ (P[:3, 3] - W.mean(0))) / np.linalg.norm(n)  # out of the wall, toward the camera
            size = np.linalg.norm(W - np.roll(W, 1, 0), axis=1).mean()
            views.setdefault(i, []).append((W.mean(0), n, size, lidar))
    out = []
    for i, vs in sorted(views.items()):
        C, N, S, L = (np.array(a) for a in zip(*vs))
        keep = np.linalg.norm(C - np.median(C, 0), axis=1) < 0.08
        if (L & keep).sum() >= 2:  # prefer the depth-lifted views
            keep &= L
        if keep.sum() < 2:
            continue  # one view (or views that disagree) is not a marker we trust
        n = N[keep].sum(0)
        out.append({"id": i, "center": np.median(C[keep], 0).round(4).tolist(),
                    "normal": (n / np.linalg.norm(n)).round(4).tolist(),
                    "size_m": round(float(np.median(S[keep])), 4), "views": int(keep.sum())})
    return out


if __name__ == "__main__":
    ex = Export(sys.argv[1])
    frames = [f for f in ex.frames if f["tracking"] == "normal"]
    found = find_markers(ex, frames, [ex.load(f)[1] for f in frames])
    print(json.dumps(found, indent=1))
    if len(sys.argv) > 2:  # compare to the synthetic ground truth
        truth = json.loads(Path(sys.argv[2]).read_text())
        H = np.array(truth["house_to_world"])
        for t in truth["markers"]:
            m = next((m for m in found if m["id"] == t["id"]), None)
            if m is None:
                print(f"id {t['id']}: not found")
                continue
            c = H[:3, :3] @ t["center"] + H[:3, 3]
            n = H[:3, :3] @ [np.cos(t["yaw"]), np.sin(t["yaw"]), 0]
            print(f"id {t['id']}: position err {100 * np.linalg.norm(m['center'] - c):.1f} cm  facing err "
                  f"{np.degrees(np.arccos(np.clip(np.dot(m['normal'], n), -1, 1))):.1f} deg  size "
                  f"{100 * m['size_m']:.1f} cm (true {100 * t['size_m']:.0f})  views {m['views']}")
