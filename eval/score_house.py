"""Build a synthetic scan with the general pipeline and score it against its truth (eval/synth_lidar.py --house multi).

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/score_house.py [out/lidar/synth-multi.zip]
Our house.json and the truth use different origins/turns, so they are lined up with the markers the pipeline found.
"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "lidar/pipeline"))
import build_house  # noqa: E402

scan = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "out/lidar/synth-multi.zip")
truth = json.loads((scan.parent / f"{scan.stem}.truth.json").read_text())
fake = lambda img, up, name: [(cls, 0.9, b) for cls, b in truth["boxes2d"].get(name, [])]  # noqa: E731 - YOLOX can't name test boxes
house, out = build_house.build(scan, ROOT / "out/houses" / scan.stem, detect_fn=fake, use_model=False)

# line up with the truth: 2D rigid fit on the markers both sides know
ours = {m["id"]: (m["x"], m["y"], m["yaw"]) for m in house["markers"]}
tru = {m["id"]: (m["center"][0], m["center"][1], m["yaw"]) for m in truth["markers"]}
ids = sorted(set(ours) & set(tru))
A, B = np.array([ours[i][:2] for i in ids]), np.array([tru[i][:2] for i in ids])
U, _, Vt = np.linalg.svd((A - A.mean(0)).T @ (B - B.mean(0)))
R = (U @ Vt).T
if np.linalg.det(R) < 0:
    R = (U @ np.diag([1, -1]) @ Vt).T
T = B.mean(0) - A.mean(0) @ R.T
to_truth = lambda p: np.asarray(p, float) @ R.T + T  # noqa: E731
turn = np.arctan2(R[1, 0], R[0, 0])
res = {"markers_found": f"{len(house['markers'])} of {len(truth['markers'])}",
       "marker_err_cm": {i: round(100 * float(np.linalg.norm(to_truth(ours[i][:2]) - tru[i][:2])), 1) for i in ids},
       "marker_yaw_err_deg": {i: round(float(np.degrees(abs((ours[i][2] + turn - tru[i][2] + np.pi) % (2 * np.pi) - np.pi))), 1) for i in ids}}

# outline: our spaces (union of rects) vs the true outer polygon, rasterised at 5 cm
C, lo = 0.05, np.array([-1.0, -1.0])
size = (int(12 / C), int(12 / C))
px = lambda pts: np.round((np.asarray(pts) - lo) / C).astype(np.int32)  # noqa: E731
true_in = cv.fillPoly(np.zeros(size, np.uint8), [px(truth["outer_polygon"])], 1)
ours_in = np.zeros(size, np.uint8)
for s in house["spaces"]:
    cv.fillPoly(ours_in, [px(to_truth(s["polygon"]))], 1)
res["outline_iou"] = round(float((true_in & ours_in).sum() / (true_in | ours_in).sum()), 3)
res["area_m2"] = {"ours": round(float(ours_in.sum() * C * C), 2), "true": round(float(true_in.sum() * C * C), 2)}

# spaces: which zones ended up sharing a space (truth: kitchen + living open plan; hallway, bedroom walled off)
res["spaces"] = [s["name"] for s in house["spaces"]]

# doors: nearest found door per true door
res["doors_found"] = f'{len(house["doors"])} (true {len(truth["doors"])})'
res["door_err_cm"] = [round(100 * min((float(np.linalg.norm(to_truth(d["center"]) - t["center"])) for d in house["doors"]), default=99), 1)
                      for t in truth["doors"]]
res["door_width_m"] = [d["width"] for d in house["doors"]]

# zones: label at every 10 cm inside each true room
hits, total = 0, 0
for r in truth["rooms"]:
    m = cv.fillPoly(np.zeros(size, np.uint8), [px(r["polygon"])], 1)
    ys, xs = np.nonzero(cv.erode(m, np.ones((5, 5), np.uint8))[::2, ::2])
    for y, x in zip(ys * 2, xs * 2):
        p = (np.array([x, y]) * C + lo - T) @ R  # truth → ours
        total += 1
        hits += build_house.zone_at([{"label": z["label"], "rects": z["rects"]} for z in house["rooms"]], p) == r["name"]
res["zone_accuracy"] = round(hits / total, 3)

# glass: is there a wall along the glass section?
def covered(a, b):
    for w in house["walls"]:
        wa, wb = to_truth(w["a"]), to_truth(w["b"])
        u = (wb - wa) / np.linalg.norm(wb - wa)
        d = lambda p: abs(u[0] * (p[1] - wa[1]) - u[1] * (p[0] - wa[0]))  # noqa: E731
        s = sorted([float(np.dot(np.asarray(a) - wa, u)), float(np.dot(np.asarray(b) - wa, u))])
        if d(a) < 0.25 and d(b) < 0.25 and s[0] > -0.3 and s[1] < np.linalg.norm(wb - wa) + 0.3:
            return True
    return False
res["glass_walled"] = [covered(g["a"], g["b"]) for g in truth["glass"]]
res["furniture"] = sorted(o["label"] for o in house["objects"])
res["landmarks"] = sorted(lm["label"] for lm in house["landmarks"])
(out / "score.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
