"""The user's zones + furniture layout (map page coords) → out/house.json (PLAN §6.3, metres, z up) + a top-view check.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python twin/make_house.py [data dir]
"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/house"
zones = json.loads((DATA / "zones.json").read_text())["zones"]
layout = json.loads((DATA / "layout.json").read_text())
markers = json.loads((DATA / "markers.json").read_text())["markers"] if (DATA / "markers.json").exists() else []
RW, RH = layout["space_m"]

house = {
    "version": 1, "units": "m", "source": layout.get("source", "v0 markerless spin (session 202610230812-g92u) + the user's zones and layout"),
    "rooms": [{"name": z["name"], "polygon": [z["from"], [z["to"][0], z["from"][1]], z["to"], [z["from"][0], z["to"][1]]]} for z in zones],
    "walls": [{"a": a, "b": b, "h": layout.get("wall_h", 2.5), "t": 0.08} for a, b in [([0, 0], [RW, 0]), ([RW, 0], [RW, RH]), ([RW, RH], [0, RH]), ([0, RH], [0, 0])]],
    "doors": layout["doors"],
    "objects": layout["objects"],
    "markers": markers,
    "landmarks": layout.get("landmarks", []),
    "accuracy": {"wall_len_err_pct_median": None, "n_checked": 0, "note": layout.get("note", "v0 model metres; counter height measured by the user")},
}
(ROOT / "out/house.json").write_text(json.dumps(house, indent=1))
(ROOT / "out/markers.csv").write_text("id,x,y,z,yaw,size_m,room\n" + "".join(
    f'{m["id"]},{m["x"]},{m["y"]},{m["z"]},{m["yaw"]},{m["size_m"]},{m["room"]}\n' for m in markers))

px = 80
img = np.full((int(RH * px) + 40, int(RW * px) + 40, 3), 255, np.uint8)
to = lambda x, y: (int(20 + x * px), int(img.shape[0] - 20 - y * px))
for z, c in zip(zones, [(60, 140, 230), (40, 120, 40), (180, 80, 160)]):
    cv.rectangle(img, to(*z["from"]), to(*z["to"]), c, 2)
    cv.putText(img, z["name"], to(z["from"][0] + 0.1, z["to"][1] - 0.3), cv.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
for f in layout["objects"]:
    box = cv.boxPoints((tuple(f["center"]), tuple(f["size"][:2]), np.degrees(f["yaw"])))
    cv.polylines(img, [np.array([to(x, y) for x, y in box], np.int32)], True, (30, 30, 200), 2)
    cv.putText(img, f["label"], to(f["center"][0] - 0.3, f["center"][1]), cv.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 200), 1)
for m in markers:
    cv.circle(img, to(m["x"], m["y"]), 7, (0, 160, 0), -1)
    cv.putText(img, f'#{m["id"]}', to(m["x"] + 0.08, m["y"] + 0.08), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 2)
cv.rectangle(img, to(0, 0), to(RW, RH), (0, 0, 0), 3)
cv.imwrite(str(ROOT / "out/house_top.png"), img)
print(f"out/house.json: {RW} x {RH} m, {len(zones)} zones, {len(layout['objects'])} pieces")
