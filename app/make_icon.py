"""Draws the app icon and the launch logo into the asset catalog. From house-twin-patrol/:
uv run --with opencv-python-headless python app/make_icon.py
"""
from pathlib import Path

import cv2 as cv
import numpy as np

ASSETS = Path(__file__).parent / "HouseTwinLidar/Assets.xcassets"
S = 2048  # drawn at 2x, then shrunk: smooth edges


def bgr(hex_):
    return tuple(int(hex_[i:i + 2], 16) for i in (5, 3, 1))


def icon():
    t = np.linspace(0, 1, S)[:, None, None]
    img = ((1 - t) * np.array(bgr("#0b2530")) + t * np.array(bgr("#0e8a94"))) * np.ones((S, S, 3))
    img = img.astype(np.uint8)
    house = np.array([[512, 1000], [1024, 520], [1536, 1000], [1536, 1560], [512, 1560]], np.int32)
    rng = np.random.default_rng(7)  # the point cloud: dots inside the house, brighter near the floor
    for _ in range(900):
        p = rng.uniform([560, 620], [1490, 1520])
        if cv.pointPolygonTest(house, p, False) > 0:
            a = 0.25 + 0.6 * (p[1] - 620) / 900
            cv.circle(img, tuple(int(v) for v in p), 7, tuple(int(c * a + 255 * (1 - a) * 0.15) for c in bgr("#7fe3e8")), -1, cv.LINE_AA)
    for a, b in [((1000, 1000), (1000, 1560)), ((512, 1240), (1000, 1240)), ((1000, 1300), (1536, 1300))]:
        cv.line(img, a, b, bgr("#bff3f5"), 26, cv.LINE_AA)  # rooms
    cv.polylines(img, [house], True, (255, 255, 255), 64, cv.LINE_AA)
    for p in house:
        cv.circle(img, tuple(int(v) for v in p), 32, (255, 255, 255), -1, cv.LINE_AA)
    cv.circle(img, (1300, 1440), 46, bgr("#ffc83d"), -1, cv.LINE_AA)  # the duck's spot on the plan
    return cv.resize(img, (1024, 1024), interpolation=cv.INTER_AREA)


if __name__ == "__main__":
    img = icon()
    cv.imwrite(str(ASSETS / "AppIcon.appiconset/icon.png"), img)
    logo = cv.cvtColor(cv.resize(img, (360, 360), interpolation=cv.INTER_AREA), cv.COLOR_BGR2BGRA)
    mask = np.zeros((1440, 1440), np.uint8)  # rounded square like the home-screen icon
    r = 320
    cv.rectangle(mask, (r, 0), (1440 - r, 1440), 255, -1)
    cv.rectangle(mask, (0, r), (1440, 1440 - r), 255, -1)
    for c in [(r, r), (1440 - r, r), (r, 1440 - r), (1440 - r, 1440 - r)]:
        cv.circle(mask, c, r, 255, -1, cv.LINE_AA)
    logo[:, :, 3] = cv.resize(mask, (360, 360), interpolation=cv.INTER_AREA)
    cv.imwrite(str(ASSETS / "LaunchLogo.imageset/logo.png"), logo)
