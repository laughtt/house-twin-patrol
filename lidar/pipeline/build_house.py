"""iPhone LiDAR export (app) → house.json (PLAN §6.3), top.png, cloud.ply, report.json. OpenCV 5.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 --with boto3 python lidar/pipeline/build_house.py <export.zip|folder> [out_dir]

Stages (twin/): scan_io → fuse → floorplan (twin/plan: floor → walls → doors → rooms) → landmarks (+ user tags) → curate (small model) → furniture → house.
No step needs a person or Claude Code; the curate model is optional (config [curate]).
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from twin import curate, floorplan, fuse, house as H, landmarks as lmk  # noqa: E402
from twin.furniture import furniture  # noqa: E402
from twin.plan.rooms import name_rooms  # noqa: E402
from twin.landmarks import upright_detect, zone_at  # noqa: E402,F401 - zone_at: eval/score_house.py
from twin.scan_io import Export  # noqa: E402


def build(src, out=None, detect_fn=upright_detect, every=2, extra_markers=None, use_model=True, extra_tags=None):
    tm, tic = {}, time.time()
    lap = lambda k: tm.__setitem__(k, time.time() - tic - sum(tm.values()))  # noqa: E731
    ex = Export(src)
    if extra_tags:
        ex.extra_tags = [{**t, "web": True} for t in json.loads(Path(extra_tags).read_text())]
    frames = [f for f in ex.frames if f["tracking"] == "normal"]
    if not frames:
        raise ValueError(f"{ex.session}: no frame with normal tracking (more light, slower walk)")
    out = Path(out or ROOT / "out" / "lidar" / ex.session)
    out.mkdir(parents=True, exist_ok=True)
    sc = fuse.fuse(ex, frames)
    lap("fuse_s")
    spaces, zones, walls_full, doors = floorplan.plan(sc.V, sc.Nv, sc.traj, sc.tlab, out)
    walls = [w_[:5] for w_ in walls_full]
    lap("rooms_s")
    cands = lmk.tagged(ex, frames, sc, walls, lmk.candidates(ex, frames, sc, zones, walls, detect_fn, every))
    near_zone = lambda xy: any(r[0] - 0.3 <= xy[0] <= r[2] + 0.3 and r[1] - 0.3 <= xy[1] <= r[3] + 0.3 for z in zones for r in z["rects"])  # noqa: E731
    cands = [c for c in cands if not c.get("removed") and (c["source"] == "user" or near_zone(c["center"]))]  # not seen through a window
    for c in cands:
        c["room"] = zone_at(zones, c["center"])
    crops = {c["id"]: lmk.crop(ex, frames, sc, c) for c in cands} if use_model else {}
    lap("landmarks_s")
    decisions, curated = curate.curate(cands, crops, [z["label"] for z in zones], use_model)
    landmarks = lmk.finish(cands, decisions)
    name_rooms(spaces, zones, landmarks)
    colours = lmk.room_colours(ex, frames, sc)
    lap("curate_s")
    objects = [ob for s in spaces for ob in furniture(sc.V, s, landmarks)]
    lap("objects_s")
    markers = H.scan_markers(ex, frames, sc, zones)
    o = np.concatenate([np.array(s["polygon"]) for s in spaces]).min(0)  # origin: lower-left of the whole house (all x, y ≥ 0)
    if extra_markers and not markers:
        markers = H.hand_markers(extra_markers, o, walls_full, objects)
    house = H.assemble(ex.session, sc, spaces, zones, walls_full, doors, objects, landmarks, markers, colours, o)
    (out / "house.json").write_text(json.dumps(house, indent=1))
    def view(c):
        k, (x0, y0, x1, y1) = c["best"]
        rw, rh = frames[k]["rgb_wh"]
        return {"i": frames[k]["i"], "uv": [round((x0 + x1) / 2 / rw, 4), round((y0 + y1) / 2 / rh, 4)]}
    (out / "candidates.json").write_text(json.dumps(
        [{k: v for k, v in c.items() if k not in ("pts", "lo", "hi", "best")} | {"center": [round(c["center"][0] - o[0], 3), round(c["center"][1] - o[1], 3)],
                                                                             "decision": decisions[c["id"]], "view": view(c)} for c in cands], indent=1))
    V = sc.V[sc.wall].copy()
    V[:, :2] -= o
    traj = sc.traj.copy()
    traj[:, :2] -= o
    H.top_view(out / "top.png", house, V, traj, [sc.names.index(r) for r in sc.tlab])
    surface = fuse.cloud(out / "cloud.ply", sc, o)
    house["occupancy"] = H.occupancy(out / "occupancy.npz", surface, np.zeros(2), (sc.gm[0] - o, sc.gm[1], sc.gm[2]))  # fused: no stray pixels
    (out / "house.json").write_text(json.dumps(house, indent=1))
    n_cloud = len(surface)
    lap("export_s")
    low = sc.traj[:, 2] < lmk.LM["low_h"]
    report = {
        "session": ex.session, "frames": len(ex.frames), "frames_used": len(frames), "frames_low": int(low.sum()),
        "voxels": int(len(sc.V)), "tsdf_points": n_cloud, "floor_offset_m": round(sc.z0, 3), "manhattan_deg": round(np.degrees(sc.theta), 2),
        "spaces": {s["name"]: {"w_m": round(float(np.ptp(np.array(s["polygon"])[:, 0])), 3), "l_m": round(float(np.ptp(np.array(s["polygon"])[:, 1])), 3),
                               "area_m2": round(float(s["mask"].sum()) * floorplan.CELL ** 2, 2), "corners": len(s["polygon"])} for s in spaces},
        "zones": {z["name"]: z["space"] for z in zones}, "doors": len(doors), "objects": len(objects),
        "landmarks": len(landmarks), "landmark_candidates": len(cands), "tags": len(ex.tags()), "curate": curated,
        "markers": len(markers), "walls": len(walls),
        "scan_frame": {"z0": sc.z0, "theta": sc.theta, "origin": [float(v) for v in o]},  # ARKit-derived world → house.json
        "timings_s": {k: round(v, 2) for k, v in tm.items()}, "cv_version": cv.__version__}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return house, out


if __name__ == "__main__":
    house, out = build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(json.loads((out / "report.json").read_text()), indent=1))
    print("wrote", out)
