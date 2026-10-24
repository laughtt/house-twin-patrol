"""Which landmark checker? The same candidates and crops of each scan go to every backend of twin/curate.py; answers
are scored against the true labels.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 --with boto3 python eval/curate_bench.py prepare <scan.zip> <name>
uv run ... python eval/curate_bench.py run [rules,opencv,clip,gemini,hybrid,haiku]   (score [...]: re-score saved answers)

prepare: the pipeline up to the landmark candidates (no checker), cached in out/curate_bench/<name>/ (cands.json,
crops/<id>.jpg, frame.json with the scan's house frame).
truth: eval/curate_truth/<name>.json {id: {"label", "keep"}} (hand-checked for our own scans), or made from an
ARKitScenes <scene>.labels.json by matching each candidate to the nearest annotated 3D box (unmatched = unknown).
A GEMINI_API_KEY in house-twin-patrol/.env is used when present.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import CFG, curate  # noqa: E402

OUT, TRUTH = ROOT / "out/curate_bench", ROOT / "eval/curate_truth"
SYN = {"hob": "stove", "cooktop": "stove", "range": "stove", "television": "tv", "monitor": "tv", "couch": "sofa", "basin": "sink",
       "refrigerator": "fridge", "plant": "potted plant", "bookcase": "bookshelf", "cupboard": "cabinet", "worktop": "kitchen counter",
       "countertop": "kitchen counter", "counter": "kitchen counter", "table": "dining table", "washing machine": "washer",
       "tub": "bathtub", "commode": "toilet", "stool": "stool", "reflection": "not a landmark", "blur": "not a landmark"}
VOCAB = sorted(set(CFG["landmarks"]["fixed"] + CFG["landmarks"]["rare"] + CFG["landmarks"]["often"] + CFG["landmarks"]["ignore"]) | set(SYN), key=len, reverse=True)


def canon(label):
    """A free-text name → the vocabulary word it contains (longest first): 'kitchen sink' → sink, 'hob (cooktop)' → stove."""
    s = str(label).lower()
    w = next((w for w in VOCAB if w in s), s)
    return SYN.get(w, w)


def prepare(src, name):
    sys.path.insert(0, str(ROOT / "lidar/pipeline"))
    from twin import floorplan, fuse, landmarks as lmk
    from twin.scan_io import Export
    ex = Export(src)
    frames = [f for f in ex.frames if f["tracking"] == "normal"]
    sc = fuse.fuse(ex, frames)
    spaces, zones, walls_full, doors = floorplan.plan(sc.V, sc.Nv, sc.traj, sc.tlab)
    walls = [w[:5] for w in walls_full]
    cands = lmk.tagged(ex, frames, sc, walls, lmk.candidates(ex, frames, sc, zones, walls))
    near = lambda xy: any(r[0] - 0.3 <= xy[0] <= r[2] + 0.3 and r[1] - 0.3 <= xy[1] <= r[3] + 0.3 for z in zones for r in z["rects"])  # noqa: E731
    cands = [c for c in cands if c["source"] == "user" or near(c["center"])]
    d = OUT / name
    (d / "crops").mkdir(parents=True, exist_ok=True)
    for c in cands:
        c["room"] = lmk.zone_at(zones, c["center"])
        cv.imwrite(str(d / "crops" / f'{c["id"]}.jpg'), lmk.crop(ex, frames, sc, c))
    keep = ("id", "cls", "label", "room", "center", "size", "yaw", "views", "low_views", "source")
    (d / "cands.json").write_text(json.dumps([{k: c[k] for k in keep} for c in cands], indent=1))
    (d / "frame.json").write_text(json.dumps({"z0": sc.z0, "theta": sc.theta, "rooms": [z["label"] for z in zones]}))
    print(f"{name}: {len(cands)} candidates → {d}")


def truth_for(name, cands):
    if (TRUTH / f"{name}.json").exists():
        return {k: v for k, v in json.loads((TRUTH / f"{name}.json").read_text()).items() if not k.startswith("_")}
    lab = ROOT / "out/arkitscenes" / f"{name}.labels.json"
    if not lab.exists():
        return {}
    fr = json.loads((OUT / name / "frame.json").read_text())
    c, s = np.cos(fr["theta"]), np.sin(fr["theta"])
    boxes = []
    for b in json.loads(lab.read_text())["boxes"]:  # ARKit world (y up) → the scan's house frame before the origin shift
        x, y = b["center"][0], -b["center"][2]
        boxes.append((canon(b["label"]), np.array([c * x + s * y, -s * x + c * y]), max(b["size"])))
    out = {}
    for cd in cands:
        best = min(boxes, key=lambda b: np.linalg.norm(b[1] - cd["center"]), default=None)
        if best is not None and np.linalg.norm(best[1] - cd["center"]) < max(0.6, best[2] / 2):
            out[cd["id"]] = {"label": best[0], "keep": True}
    return out


def score(dec, truth):
    ids = [i for i in truth if i in dec]
    if not ids:
        return {}
    ok = lambda i: canon(dec[i]["label"]) in {canon(w) for w in [truth[i]["label"], *truth[i].get("also", [])]}  # noqa: E731
    lab = [ok(i) for i in ids if truth[i]["keep"]]
    keep = [dec[i]["keep"] == truth[i]["keep"] for i in ids]
    mob = [dec[i]["mobility"] == curate.mobility(canon(truth[i]["label"])) for i in ids if truth[i]["keep"]]
    return {"label_ok": f"{sum(lab)}/{len(lab)}", "keep_ok": f"{sum(keep)}/{len(keep)}", "mobility_ok": f"{sum(mob)}/{len(mob)}",
            "label_pct": round(100 * np.mean(lab), 1) if lab else None, "keep_pct": round(100 * np.mean(keep), 1)}


def run(backends):
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                os.environ.setdefault(*line.split("=", 1))
    rows = []
    for d in sorted(p for p in OUT.iterdir() if (p / "cands.json").exists()):
        cands = json.loads((d / "cands.json").read_text())
        crops = {c["id"]: cv.imread(str(d / "crops" / f'{c["id"]}.jpg')) for c in cands}
        truth = truth_for(d.name, cands)
        rooms = json.loads((d / "frame.json").read_text())["rooms"]
        for b in backends:
            t = time.time()
            if SCORE_ONLY:  # re-score the saved answers (no new API calls)
                saved = json.loads((d / f"decisions_{b}.json").read_text())
                dec, note = saved["decisions"], {**saved["note"], "seconds": saved["note"].get("seconds", 0)}
            else:
                dec, note = curate.curate(cands, crops, rooms, backend=b)
            row = {"map": d.name, "backend": b, "candidates": len(cands), "with_truth": len(truth), "seconds": note.get("seconds", 0) if SCORE_ONLY else round(time.time() - t, 1),
                   "sent_to_cloud": len(cands) if b in ("gemini", "haiku") else int(str(note.get("asked", "0/")).split("/")[0] or 0),
                   **score(dec, truth), "error": note.get("error", "")[:80]}
            rows.append(row)
            if not SCORE_ONLY:
                (d / f"decisions_{b}.json").write_text(json.dumps({"note": note, "decisions": dec}, indent=1))
            print(json.dumps(row))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    SCORE_ONLY = sys.argv[1] == "score"
    if sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[2].split(",") if len(sys.argv) > 2 else ["rules", "opencv", "clip", "gemini", "hybrid"])
