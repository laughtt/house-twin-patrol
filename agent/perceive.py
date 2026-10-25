"""OpenCV 5 on the duck camera: ArUco markers, recognised objects (bearing + range) and the room's colour fingerprint."""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), cv.aruco.DetectorParameters())


def markers(gray, K, sizes):
    """[{id, corners, solutions: [(R_cm, t_cm, reproj_px), ...]}] for every known marker in view (sizes: id → metres)."""
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        if int(i) not in sizes:
            continue
        s = sizes[int(i)] / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        _, rvecs, tvecs, errs = cv.solvePnPGeneric(obj, c[0], K, None, flags=cv.SOLVEPNP_IPPE_SQUARE)
        out.append({"id": int(i), "corners": c[0],
                    "solutions": [(cv.Rodrigues(r)[0], t.ravel(), float(e)) for r, t, e in zip(rvecs, tvecs, np.ravel(errs))]})
    return out


def objects(bgr, K, eye_h, min_score=0.4):
    """YOLOX (the scan's detector) on an upright, level camera → [{cls, score, bearing (rad, + = left), range (m) or
    None}]. Range comes from where the box meets the floor, so only when its bottom edge is in view."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
    from v0.scan import LANDMARK, detect
    h = bgr.shape[0]
    out = []
    for cls, score, (x0, y0, x1, y1) in detect(bgr, classes=LANDMARK):
        if score < min_score:
            continue
        below = np.arctan2(y1 - K[1, 2], K[1, 1])
        rng = float(eye_h / np.tan(below)) if y1 < h - 3 and below > np.radians(3) else None
        out.append({"cls": cls, "score": float(score), "bearing": float(np.arctan2(K[0, 2] - (x0 + x1) / 2, K[0, 0])), "range": rng})
    return out


def colour_hist(bgr):
    """Hue × saturation histogram (16 × 8) of the well-lit pixels, summing to 1: a room's colour fingerprint."""
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (0, 0, 40), (180, 255, 255))
    hist = cv.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256]).ravel()
    return hist / max(hist.sum(), 1)


def see(pf, bgr, K, max_range=3.0):
    """OpenCV on one camera frame, into the filter: the room's colour (a weak vote), then each ArUco marker on the map
    within max_range (a snap; a farther one is a few pixels across and its pose can be metres off).
    → (marker ids used, the room the colours look most like)."""
    hist = colour_hist(bgr)
    pf.update_room(hist)
    sims = [np.sum(np.sqrt(hist * r["colour_hs"])) if "colour_hs" in r else -1 for r in pf.map.rooms]
    ids = []
    for m in markers(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY), K, {i: m["size_m"] for i, m in pf.map.markers.items()}):
        if np.linalg.norm(m["solutions"][0][1]) < max_range:
            pf.update_marker(m["id"], m["solutions"])
            ids.append(m["id"])
    return ids, pf.map.rooms[int(np.argmax(sims))]["name"] if max(sims, default=-1) >= 0 else None
