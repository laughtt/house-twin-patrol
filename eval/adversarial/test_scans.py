"""Whole pipeline (lidar/pipeline/build_house.py, no model) on hostile variants of the synthetic scan
out/lidar/synth-multi.zip (make it with: eval/synth_lidar.py --house multi). Detector = the true 2D boxes. ~5 s a build."""
import json
import sys
import zipfile

import numpy as np
import pytest

import sim

ZIP, TRUTH = sim.ROOT / "out/lidar/synth-multi.zip", sim.ROOT / "out/lidar/synth-multi.truth.json"
sys.path.insert(0, str(sim.ROOT / "lidar/pipeline"))
pytestmark = pytest.mark.skipif(not ZIP.exists(), reason="run eval/synth_lidar.py --house multi first")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(root)
    return root / "synth-multi"


@pytest.fixture(scope="module")
def truth():
    return json.loads(TRUTH.read_text())


def variant(base, tmp, frames=None, labels=None, events="keep"):
    tmp.mkdir()
    for sub in ("rgb", "depth", "conf"):
        (tmp / sub).symlink_to(base / sub)
    (tmp / "session.json").write_text((base / "session.json").read_text())
    rows = [json.loads(s) for s in (base / "frames.jsonl").read_text().splitlines() if s.strip()]
    (tmp / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in (frames(rows) if frames else rows)))
    if events == "keep":
        (tmp / "events.jsonl").write_text((base / "events.jsonl").read_text())
    if labels is not None:
        (tmp / "labels.json").write_text(json.dumps(labels))
    return tmp


def build(src, truth, detect=None):
    import build_house
    fake = detect or (lambda img, up, name: [(cls, 0.9, b) for cls, b in truth["boxes2d"].get(name, [])])
    try:
        house, out = build_house.build(src, src.parent / (src.name + "_out"), detect_fn=fake, use_model=False)
        return house, json.loads((out / "report.json").read_text()), None
    except Exception as e:  # noqa: BLE001
        return None, None, e


def test_zero_detections(base, truth, tmp_path):
    house, rep, err = build(variant(base, tmp_path / "v"), truth, detect=lambda *a: [])
    ok = err is None and house["landmarks"] == [] and len(house["rooms"]) >= 3
    sim.report("scan_zero_detections", ok, f"error {err!r}" if err else f"{len(house['rooms'])} rooms, {len(house['landmarks'])} landmarks")


def test_all_tracking_limited(base, truth, tmp_path):
    """Every frame 'limited': nothing to build from. Pass = a clear message (names frames/tracking), not a deep crash."""
    house, rep, err = build(variant(base, tmp_path / "v", frames=lambda fs: [f | {"tracking": "limited"} for f in fs]), truth)
    clear = err is not None and any(w in str(err).lower() for w in ("frame", "tracking"))
    sim.report("scan_all_tracking_limited", clear, f"raised {type(err).__name__}: {str(err)[:90]!r}" if err else "built a house from no frames?")


def test_half_tracking_limited(base, truth, tmp_path):
    house, rep, err = build(variant(base, tmp_path / "v", frames=lambda fs: [f | {"tracking": "limited"} if f["i"] % 2 else f for f in fs]), truth)
    n = len(house["landmarks"]) if house else 0
    sim.report("scan_half_tracking_limited", err is None and n >= 4, f"error {err!r}" if err else f"{n}/5 landmarks")


def test_no_events_no_labels(base, truth, tmp_path):
    house, rep, err = build(variant(base, tmp_path / "v", events=None), truth)
    sim.report("scan_no_events_no_labels", err is None, f"error {err!r}" if err else f"{len(house['landmarks'])} landmarks, {len(house['rooms'])} rooms")


def test_no_low_frames_room_colours(base, truth, tmp_path):
    """The synthetic walk is all chest height (1.35 m): no duck-height frames. Colours must still exist for every room."""
    house, rep, err = build(variant(base, tmp_path / "v"), truth)
    missing = [r["name"] for r in house["rooms"] if len(r.get("colour_hs", [])) != 128] if house else ["<no house>"]
    sim.report("scan_no_low_frames", err is None and rep["frames_low"] == 0 and not missing, f"low frames {rep and rep['frames_low']}, rooms without colour {missing}")


def _fridge_frames(truth):
    return [(int(name), [b for cls, b in boxes if cls == "refrigerator"][0]) for name, boxes in sorted(truth["boxes2d"].items())
            if any(cls == "refrigerator" for cls, _ in boxes)]


def _empty_spot(base):
    """A frame and crosshair with no depth in a 33 px window (looking out through the glass door)."""
    rows = [json.loads(s) for s in (base / "frames.jsonl").read_text().splitlines() if s.strip()]
    for f in rows[::3]:
        w, h = f["depth_wh"]
        d = np.frombuffer((base / f["depth"]).read_bytes(), np.float16).reshape(h, w).astype(float)
        c = np.frombuffer((base / f["conf"]).read_bytes(), np.uint8).reshape(h, w)
        ok = (c >= 1) & (d > 0.2) & (d < 5.0)
        for v in range(20, h - 20, 8):
            for u in range(20, w - 20, 8):
                if not ok[v - 17:v + 18, u - 17:u + 18].any():
                    return f["i"], [u / w, v / h]
    return None


@pytest.fixture(scope="module")
def tagged_house(base, truth, tmp_path_factory):
    """labels.json tags: the fridge tagged twice from two frames, a tag on empty space, a tag on a frame index that
    does not exist, a tag with the crosshair off the picture."""
    (i1, b1), (i2, b2) = _fridge_frames(truth)[0], _fridge_frames(truth)[-1]
    uv = lambda b: [(b[0] + b[2]) / 2 / 960, (b[1] + b[3]) / 2 / 720]  # noqa: E731
    spot = _empty_spot(base)
    tags = [{"i": i1, "label": "Fridge", "uv": uv(b1)}, {"i": i2, "label": "fridge", "uv": uv(b2)},
            {"i": 10 ** 6, "label": "plant", "uv": [0.5, 0.5]}, {"i": i1, "label": "lamp", "uv": [-0.4, 0.5]}]
    if spot:
        tags.append({"i": spot[0], "label": "tv", "uv": spot[1]})
    house, rep, err = build(variant(base, tmp_path_factory.mktemp("tags") / "v", labels={"tags": tags}), truth)
    return house, err, spot


def test_tags_end_to_end(tagged_house):
    """Pass = one fridge landmark, labelled by the user, the empty-space tag dropped, nothing outside the house."""
    house, err, spot = tagged_house
    if err:
        return sim.report("scan_tags_end_to_end", False, f"error {err!r}")
    fr = [lm for lm in house["landmarks"] if lm["label"] == "fridge"]
    xs = [p for w in house["walls"] for p in (w["a"][0], w["b"][0])]
    ys = [p for w in house["walls"] for p in (w["a"][1], w["b"][1])]
    outside = [(lm["label"], lm["center"]) for lm in house["landmarks"]
               if not (min(xs) - 0.1 <= lm["center"][0] <= max(xs) + 0.1 and min(ys) - 0.1 <= lm["center"][1] <= max(ys) + 0.1)]
    ok = len(fr) == 1 and fr[0]["source"] == "user" and not outside and "tv" not in [lm["label"] for lm in house["landmarks"]]
    sim.report("scan_tags_end_to_end", ok, f"fridge landmarks {[(lm['source'], lm['center']) for lm in fr]}, outside the walls {outside}, "
                                           f"labels {sorted(lm['label'] for lm in house['landmarks'])}, empty-space tag {'tried' if spot else 'no spot found'}")


def test_tag_on_missing_frame(tagged_house):
    """A tag whose frame index is nowhere near any frame (1e6) must not name whatever the nearest frame shows."""
    house, err, _ = tagged_house
    labels = [lm["label"] for lm in house["landmarks"]] if house else []
    sim.report("scan_tag_frame_missing", err is None and "plant" not in labels,
               f"error {err!r}" if err else "tag on frame 1e6 " + ("became a 'plant' landmark from the last frame's centre" if "plant" in labels else "was dropped"))
