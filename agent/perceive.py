"""OpenCV 5 on the duck camera: ArUco markers → both IPPE solutions of the camera pose in each marker's frame."""
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
