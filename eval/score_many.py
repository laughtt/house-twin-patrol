"""Build every generated house (eval/gen_houses.py) with the general pipeline and score it against its truth.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/score_many.py [name ...]
→ out/synth_houses/scores.json, scores.md; builds in out/synth_houses/<name>/.
Alignment: the truth's house_to_world and report.json's scan_frame (theta, origin) give the exact map from house.json
to the truth frame (no markers needed). A crash is recorded with its traceback and the next house goes on.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "lidar/pipeline"))
import build_house  # noqa: E402
from twin.landmarks import name_for  # noqa: E402

DIR = ROOT / "out" / "synth_houses"
C = 0.05


def score(zpath):
    truth = json.loads(zpath.with_suffix(".truth.json").read_text())
    fake = lambda img, up, name: [(cls, 0.9, b) for cls, b in truth["boxes2d"].get(name, [])]  # noqa: E731
    tic = time.time()
    house, out = build_house.build(zpath, DIR / zpath.stem, detect_fn=fake, use_model=False)
    res = {"build_s": round(time.time() - tic, 1)}
    rep = json.loads((out / "report.json").read_text())
    th, o = rep["scan_frame"]["theta"], np.array(rep["scan_frame"]["origin"])
    R2 = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    Hm = np.array(truth["house_to_world"])
    to_truth = lambda p: ((np.asarray(p, float) + o) @ R2 - Hm[:2, 3]) @ Hm[:2, :2]  # noqa: E731
    to_ours = lambda p: ((np.asarray(p, float) @ Hm[:2, :2].T + Hm[:2, 3]) @ R2.T) - o  # noqa: E731

    allp = np.array(truth["outer_polygon"])
    lo = allp.min(0) - 2
    size = tuple(np.ceil((allp.max(0) + 2 - lo) / C).astype(int)[::-1])
    px = lambda pts: np.round((np.asarray(pts) - lo) / C).astype(np.int32)  # noqa: E731
    true_in = cv.fillPoly(np.zeros(size, np.uint8), [px(allp)], 1)
    ours_in = np.zeros(size, np.uint8)
    for s in house["spaces"]:
        cv.fillPoly(ours_in, [px(to_truth(s["polygon"]))], 1)
    res["iou"] = round(float((true_in & ours_in).sum() / (true_in | ours_in).sum()), 3)
    res["area_err_pct"] = round(100 * float(ours_in.sum() / true_in.sum() - 1), 1)
    res["spaces"] = f'{len(house["spaces"])}/{len(truth["spaces"])}'
    res["space_names"] = [s["name"] for s in house["spaces"]]

    pairs = []  # (centre error, width error) of the nearest found door per true door
    for t in truth["doors"]:
        d = min(house["doors"], key=lambda d: float(np.linalg.norm(to_truth(d["center"]) - t["center"])), default=None)
        if d is not None and np.linalg.norm(to_truth(d["center"]) - t["center"]) < 0.5:
            pairs.append((float(np.linalg.norm(to_truth(d["center"]) - t["center"])), d["width"] - t["width"]))
    res["doors"] = f'{len(pairs)}/{len(truth["doors"])}' + (f' (+{len(house["doors"]) - len(pairs)} extra)' if len(house["doors"]) > len(pairs) else "")
    res["door_err_cm"] = round(100 * float(np.mean([p[0] for p in pairs])), 1) if pairs else None
    res["door_width_err_cm"] = round(100 * float(np.mean([p[1] for p in pairs])), 1) if pairs else None

    named = [r for r in truth["rooms"] if r["zone"]]
    if len(truth["zones"]) > 1:
        hits = total = 0
        rooms = [{"label": z["label"], "rects": z["rects"]} for z in house["rooms"]]
        for r in named:
            m = cv.erode(cv.fillPoly(np.zeros(size, np.uint8), [px(r["polygon"])], 1), np.ones((5, 5), np.uint8))
            ys, xs = np.nonzero(m[::2, ::2])
            pts = to_ours(np.c_[xs * 2, ys * 2] * C + lo + C / 2)
            total += len(pts)
            hits += sum(build_house.zone_at(rooms, p) == r["zone"] for p in pts)
        res["zone_acc"] = round(hits / max(total, 1), 3)
    else:
        res["zone_acc"] = None

    def covered(a, b):  # ≥ 90 % of the glass line lies within 25 cm of some wall (walls may be split along it)
        segs = [(to_truth(w["a"]), to_truth(w["b"])) for w in house["walls"]]
        def near(p):
            for wa, wb in segs:
                u = wb - wa
                t = np.clip(np.dot(p - wa, u) / max(np.dot(u, u), 1e-9), 0, 1)
                if np.linalg.norm(wa + t * u - p) < 0.25:
                    return True
            return False
        pts = np.linspace(a, b, 30)
        return np.mean([near(p) for p in pts]) >= 0.9
    res["glass_sealed"] = f'{sum(covered(g["a"], g["b"]) for g in truth["glass"])}/{len(truth["glass"])}' if truth["glass"] else None

    def footprint(ob):  # found object → its axis-aligned box in the truth frame
        c, s, yaw = ob["center"], ob["size"], ob.get("yaw", 0.0)
        q = to_truth(cv.boxPoints((tuple(c), tuple(s[:2]), np.degrees(yaw))))
        return q.min(0), q.max(0)
    found = [footprint(ob) for ob in house["objects"]]
    floor = [f for f in truth["furniture"] if f["lo"][2] - truth["rooms"][[r["name"] for r in truth["rooms"]].index(f["room"])]["floor_z"] < 0.1]
    def overlap(f, b):
        w = np.clip(np.minimum(f["hi"][:2], b[1]) - np.maximum(f["lo"][:2], b[0]), 0, None)
        return w.prod() / np.prod(np.subtract(f["hi"][:2], f["lo"][:2]))
    got = [f for f in floor if any(overlap(f, b) >= 0.25 for b in found)]
    res["furniture"] = f"{len(got)}/{len(floor)}"
    res["furniture_touched"] = f'{sum(any(overlap(f, b) > 0 for b in found) for f in floor)}/{len(floor)}'
    res["furniture_missed"] = sorted(f["label"] for f in floor if f not in got)

    zone = {r["name"]: r["zone"] or "" for r in truth["rooms"]}
    lms = [(lm["label"], to_truth(lm["center"])) for lm in house["landmarks"]]
    by = {}
    for t in truth["landmarks"]:
        want = name_for(t["cls"], zone[t["room"]])
        if not want:
            continue
        half = np.array(t["size"][:2]) / 2 + 0.4
        ok = any(lab == want and (np.abs(c - t["center"][:2]) <= half).all() for lab, c in lms)
        k = by.setdefault(want, [0, 0])
        k[0] += ok
        k[1] += 1
    res["landmarks"] = f'{sum(v[0] for v in by.values())}/{sum(v[1] for v in by.values())}'
    res["landmarks_by_class"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by.items())}
    res["landmarks_found"] = len(house["landmarks"])
    res["frames_low"] = f'{rep["frames_low"]}/{truth["frames_low"]}'
    res["frames"] = rep["frames"]
    return res


def table(rows):
    cols = ["house", "frames", "build_s", "iou", "area_err_pct", "spaces", "doors", "door_err_cm", "door_width_err_cm", "zone_acc", "glass_sealed",
            "furniture", "furniture_touched", "landmarks", "frames_low"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, r in rows.items():
        out.append("| " + " | ".join([name] + (["CRASH: " + r["error"].splitlines()[-1][:80]] + [""] * (len(cols) - 2) if "error" in r
                                                else ["" if r.get(c) is None else str(r[c]) for c in cols[1:]])) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    rows = {}
    for z in sorted(DIR.glob("*.zip")):
        if want and z.stem not in want:
            continue
        try:
            rows[z.stem] = score(z)
        except Exception:
            rows[z.stem] = {"error": traceback.format_exc(), "input": str(z)}
        print(z.stem, json.dumps({k: v for k, v in rows[z.stem].items() if k != "error"}), flush=True)
        if "error" in rows[z.stem]:
            print(rows[z.stem]["error"][-1500:], flush=True)
    path = DIR / ("scores.json" if not want else "scores_subset.json")
    path.write_text(json.dumps(rows, indent=1))
    path.with_suffix(".md").write_text(table(rows) + "\n")
    print(table(rows))
