"""Whole-route adversarial cases on the real LiDAR house (and a symmetric twin-room house), fast simulator (sim.py).
Every case runs ADV_SEEDS seeds (default 4; 8 for the full report) in one process pool, then each test checks its
numbers. Criteria are written before running and are not tuned to pass."""
import copy
import math
import os

import numpy as np
import pytest

import sim
from sim import HFOV, HouseMap, real_house, truth_path, wrap

SEEDS = list(range(int(os.environ.get("ADV_SEEDS", 4))))
TP = truth_path()
H = real_house()
CLASSES = sorted({lm["cls"] for lm in H["landmarks"] if lm.get("cls") and lm.get("weight", 0) > 0})
LMS = [lm for lm in H["landmarks"] if lm.get("cls") and lm.get("weight", 0) > 0]


def need(frac):
    return math.ceil(frac * len(SEEDS) - 1e-9)


# ---- standard criteria ------------------------------------------------------------------------------------------
def std(errs, lock_by=20.0, median=0.15, wrong_max=None):
    """Finds itself (0.3 m held 2 s) within lock_by s in ≥ 75 % of seeds, median error after lock ≤ median;
    optionally ≤ wrong_max of the after-lock frames more than 1 m off (confidently in the wrong place)."""
    locks = [sim.lock(e) for e in errs]
    ok_seeds = [t is not None and t <= lock_by for t in locks]
    meds = [sim.after_lock_median(e) for e, ok in zip(errs, ok_seeds) if ok]
    med = float(np.median(meds)) if meds else float("inf")
    wrong = [float(np.mean(e[round(t * sim.HZ):] > 1.0)) for e, t in zip(errs, locks) if t is not None]
    worst_wrong = max(wrong) if wrong else 1.0
    passed = sum(ok_seeds) >= need(0.75) and med <= median and (wrong_max is None or worst_wrong <= wrong_max)
    detail = (f"locked {sum(ok_seeds)}/{len(errs)} (≤{lock_by:.0f}s, need {need(0.75)}), lock s {locks}, "
              f"after-lock median {med:.3f} m (≤{median})" + (f", worst >1 m share {worst_wrong:.2f} (≤{wrong_max})" if wrong_max is not None else ""))
    return passed, detail


def relock(errs, at, within_s, frac=0.75):
    """After frame index `at` (kidnap, glitch) the error is back under 0.3 m (held 2 s) within `within_s` s."""
    t = [sim.lock(e, at) for e in errs]
    ok = sum(x is not None and x <= within_s for x in t)
    return ok >= need(frac), f"re-locked within {within_s}s in {ok}/{len(errs)} (need {need(frac)}), seconds {t}"


def holds(errs, a, b=None, share=0.8, frac=0.75):
    """Within 0.3 m in ≥ share of frames a..b, in ≥ frac of seeds (known start cases)."""
    w = [round(sim.within(e, a, b), 2) for e in errs]
    ok = sum(x >= share for x in w)
    return ok >= need(frac), f"within 0.3 m share per seed {w} (need ≥{share} in {need(frac)})"


# ---- adversarial inputs --------------------------------------------------------------------------------------------
def visible(world, pose, pt, r_max=4.5):
    x, y, yaw = pose
    r = math.hypot(pt[0] - x, pt[1] - y)
    b = wrap(math.atan2(pt[1] - y, pt[0] - x) - yaw)
    clear = world.raycast(np.array([x]), np.array([y]), np.array([yaw + b]))[0] >= min(r - 0.2, 4)
    return (b if r < r_max and abs(b) < HFOV / 2 and clear else None), r


def fp_burst(seed):
    r = np.random.default_rng(100 + seed)

    def t(k, pose, odo, dets, tof):
        if 100 <= k < 200:  # someone walks around the duck for 20 s: 3 junk sightings per frame, any class
            dets = dets + [{"cls": str(r.choice(CLASSES)), "bearing": float(r.uniform(-HFOV / 2, HFOV / 2)),
                            "range": float(r.uniform(0.5, 3)) if r.random() < 0.5 else None} for _ in range(3)]
        return odo, dets, tof
    return t


def relabel(mapping):
    def t(k, pose, odo, dets, tof):
        return odo, [dict(d, cls=mapping.get(d["cls"], d["cls"])) for d in dets], tof
    return t


def tv_fridge(seed):
    """A TV on the living room's north wall shows a fridge advert all the time (a fixed-class phantom)."""
    r, world = np.random.default_rng(200 + seed), HouseMap(H)

    def t(k, pose, odo, dets, tof):
        b, _ = visible(world, pose, (3.0, 6.7))
        if b is not None and r.random() < 0.7:
            dets = dets + [{"cls": "refrigerator", "bearing": b + r.normal(0, math.radians(3)), "range": None}]
        return odo, dets, tof
    return t


def stuck_phantom(k, pose, odo, dets, tof):
    return odo, [{"cls": "refrigerator", "bearing": 0.0, "range": 1.0}], tof


def tof_from(k0, kind):
    held = {}

    def t(k, pose, odo, dets, tof):
        if k < k0:
            return odo, dets, tof
        dist, status = tof
        if kind == "no_data":
            status = np.full(64, 254)
        elif kind == "no_target":
            status = np.full(64, 255)
        elif kind == "covered":
            dist, status = np.full(64, 15.0), np.full(64, 5)
        elif kind == "stuck":
            held.setdefault("f", (dist, status))
            dist, status = held["f"]
        return odo, dets, (dist, status)
    return t


def at_frame(k0, fn):
    def t(k, pose, odo, dets, tof):
        return (fn(odo) if k == k0 else odo), dets, tof
    return t


SLIP = (120, 145)  # duck stuck against the kitchen wall for 5 s, wheels still turning


def slip(k, pose, odo, dets, tof):
    return ((0.052, 0.001, 0.0) if SLIP[0] <= k < SLIP[1] else odo), dets, tof


def moved(fn):
    out = copy.deepcopy(LMS)
    for i, lm in enumerate(out):
        fn(i, lm)
    return out


def shuffle_movables(i, lm):
    a = float(np.random.default_rng(11 + i).uniform(-math.pi, math.pi))
    if lm["mobility"] in ("often", "rare") and lm["cls"] != "couch":
        lm["center"] = [lm["center"][0] + math.cos(a), lm["center"][1] + math.sin(a)]


def shift_fixed(i, lm):
    a = float(np.random.default_rng(31 + i).uniform(-math.pi, math.pi))
    if lm["mobility"] == "fixed":
        lm["center"] = [lm["center"][0] + math.cos(a), lm["center"][1] + math.sin(a)]


def lookalikes():
    h = copy.deepcopy(H)
    chair = {"label": "chair", "cls": "chair", "size": [0.5, 0.5, 0.9], "yaw": 0.0, "mobility": "often", "weight": 0.2}
    counter = {"label": "counter", "cls": "sink", "size": [0.6, 0.6, 0.9], "yaw": 0.0, "mobility": "fixed", "weight": 1.0}
    h["landmarks"] += [dict(chair, center=c) for c in ([0.6, 3.0], [2.4, 3.0], [0.7, 5.6], [2.6, 5.6], [2.2, 1.4], [0.8, 1.4])]
    h["landmarks"] += [dict(counter, center=c) for c in ([0.35, 1.6], [3.55, 3.4], [0.35, 5.2])]
    return h


# ---- room colour ---------------------------------------------------------------------------------------------------
def colour(mode, seed):
    r = np.random.default_rng(300 + seed)

    def hist(k, pose, world):
        i = int(world.room(pose[0], pose[1]))
        if i < 0 or "colour_hs" not in world.rooms[i]:
            return None
        if mode == "wrong":
            i = (i + 1) % len(world.rooms)
        h = np.asarray(world.rooms[i]["colour_hs"]) * r.gamma(20, 1 / 20, 128)
        if mode == "lights":  # warmer bulb / evening: hue shifted two bins
            h = np.roll(h.reshape(16, 8), 2, axis=0).ravel()
        return h / h.sum()
    return hist


def same_colours():
    h = copy.deepcopy(H)
    for room in h["rooms"]:
        room["colour_hs"] = H["rooms"][0]["colour_hs"]
    return h


def some_colours_missing():
    h = copy.deepcopy(H)
    for room in h["rooms"][:2]:
        del room["colour_hs"]
    return h


# ---- the symmetric house: two identical rooms, 180° rotation symmetry --------------------------------------------
def twin_house(tv=False, colours=False):
    W, L = 6.1, 3.0
    rot = lambda p: [round(W - p[0], 3), round(L - p[1], 3)]  # noqa: E731
    walls = [((0, 0), (W, 0)), ((W, 0), (W, L)), ((W, L), (0, L)), ((0, L), (0, 0)), ((3.05, 0), (3.05, 1.05)), ((3.05, 1.95), (3.05, L))]
    objs = [("counter", "sink", [0.3, 2.3], [0.6, 1.2, 0.9], "fixed"), ("fridge", "refrigerator", [2.6, 2.65], [0.6, 0.6, 1.8], "fixed"),
            ("sofa", "couch", [1.0, 0.4], [1.8, 0.8, 0.85], "rare")]
    h = {"walls": [{"a": list(a), "b": list(b), "t": 0.1} for a, b in walls], "objects": [], "landmarks": [], "markers": []}
    for label, cls, c, s, mob in objs:
        for cc in (c, rot(c)):
            h["objects"].append({"label": label, "kind": "box", "center": cc, "size": s, "yaw": 0.0})
            h["landmarks"].append({"label": label, "cls": cls, "center": cc, "size": s, "yaw": 0.0, "mobility": mob,
                                   "weight": {"fixed": 1.0, "rare": 0.6}[mob]})
    if tv:  # the one thing that tells the rooms apart: a TV on room A's north wall
        h["landmarks"].append({"label": "tv", "cls": "tv", "center": [1.6, 2.95], "size": [1.0, 0.1, 0.6], "yaw": 0.0, "mobility": "rare", "weight": 0.6})
    g = np.random.default_rng(5)
    ca, cb = (g.dirichlet(np.full(128, 0.3)).round(5).tolist() for _ in range(2))
    h["rooms"] = [{"name": "a", "rects": [[0.05, 0.05, 3.0, 2.95]], "colour_hs": ca},
                  {"name": "b", "rects": [[3.1, 0.05, 6.05, 2.95]], "colour_hs": cb if colours else ca}]
    return h


TWIN_TRUTH = truth_path([(1.0, 1.2), (2.2, 1.2), ("face", math.pi / 2), (2.2, 2.0), (1.0, 2.0), (1.0, 1.2), (2.2, 1.2), (2.2, 2.0), ("face", math.pi / 2)])


def twin_share(result):
    """(errors, weight share in the true room A, in the twin room B) at the end of the run."""
    err, pf = result
    room = pf.map.room(pf.p[:, 0], pf.p[:, 1])
    return err, float(pf.w[room == 0].sum()), float(pf.w[room == 1].sum())


# ---- the case table: name → (task(seed), check(results)) ------------------------------------------------------------
K0 = 150  # kidnap frame: the duck has been tracking for 30 s in the kitchen
CASES = {
    "baseline": (lambda s: sim.simulate(s), std),
    # 1 kidnapping (carried, odometry says nothing happened)
    "kidnap_other_room": (lambda s: sim.simulate(s, truth=TP[:K0] + TP[330:], jumps={K0}), lambda e: relock(e, K0 - 1, 15)),
    "kidnap_same_room": (lambda s: sim.simulate(s, truth=TP[:K0] + TP[240:], jumps={K0}), lambda e: relock(e, K0 - 1, 15)),
    "kidnap_lights_off": (lambda s: sim.simulate(s, truth=TP[:K0] + TP[240:], jumps={K0}, det_kw={"hit": 0, "false": 0}),
                          lambda e: relock(e, K0 - 1, 15)),
    # 2 symmetry and look-alikes
    "twin_rooms_no_cue": (lambda s: sim.simulate(s, house=twin_house(), truth=TWIN_TRUTH, ret_pf=True),
                          lambda r: (lambda sh: (sum(0.1 <= b for _, a, b in sh) >= need(0.75),
                                                 f"twin room B keeps ≥10 % of the weight in {sum(0.1 <= b for _, a, b in sh)}/{len(sh)} "
                                                 f"(need {need(0.75)}); shares A/B {[(round(a, 2), round(b, 2)) for _, a, b in sh]}"))([twin_share(x) for x in r])),
    "twin_rooms_tv_cue": (lambda s: sim.simulate(s, house=twin_house(tv=True), truth=TWIN_TRUTH), lambda e: holds(e, -50, None, 0.9)),
    "twin_rooms_colour_cue": (lambda s: sim.simulate(s, house=twin_house(colours=True), truth=TWIN_TRUTH, room_hist=colour("true", s)),
                              lambda e: holds(e, -50, None, 0.9)),
    "lookalikes_6chairs_8counters": (lambda s: sim.simulate(s, house=lookalikes()), std),
    # 3 moved / removed landmarks (the map is from scan day, the world is today)
    "fridge_removed": (lambda s: sim.simulate(s, world_lms=[lm for lm in LMS if lm["cls"] != "refrigerator"]), std),
    "sofa_moved_2m": (lambda s: sim.simulate(s, world_lms=moved(lambda i, lm: lm["cls"] == "couch" and lm.update(center=[lm["center"][0], lm["center"][1] + 2.0]))), std),
    "movables_shuffled_1m": (lambda s: sim.simulate(s, world_lms=moved(shuffle_movables)), std),
    "fixed_fridge_moved_3m": (lambda s: sim.simulate(s, world_lms=moved(lambda i, lm: lm["cls"] == "refrigerator" and lm.update(center=[0.4, 1.6]))),
                              lambda e: std(e, wrong_max=0.05)),
    "all_fixed_moved_1m": (lambda s: sim.simulate(s, world_lms=moved(shift_fixed)), lambda e: std(e, median=0.2)),
    # 4 the detector lies
    "fp_burst_person_20s": (lambda s: sim.simulate(s, tamper=fp_burst(s)),
                            lambda e: (lambda a, b: (a[0] and b[0], a[1] + " | burst: " + b[1]))(std(e), holds(e, 99, 220, 0.85))),
    "couch_table_swapped": (lambda s: sim.simulate(s, tamper=relabel({"couch": "dining table", "dining table": "couch"})), std),
    "couch_called_bed": (lambda s: sim.simulate(s, tamper=relabel({"couch": "bed"})), std),
    "tv_shows_fridge": (lambda s: sim.simulate(s, tamper=tv_fridge(s)), lambda e: std(e, wrong_max=0.05)),
    "bad_detector": (lambda s: sim.simulate(s, det_kw={"hit": 0.5, "wrong": 0.3, "false": 0.5}), lambda e: std(e, median=0.2)),
    "lights_off_unknown_start": (lambda s: sim.simulate(s, det_kw={"hit": 0, "false": 0}), std),
    "lights_off_known_start": (lambda s: sim.simulate(s, start="known", det_kw={"hit": 0, "false": 0}), lambda e: holds(e, 0, None, 0.9)),
    "stuck_phantom_fridge": (lambda s: sim.simulate(s, start="known", tamper=stuck_phantom), lambda e: holds(e, 0, None, 0.8)),
    # 5 sensor failures
    "tof_no_data_all_254": (lambda s: sim.simulate(s, tamper=tof_from(0, "no_data")), lambda e: std(e, median=0.3)),
    "tof_no_target_from_20s": (lambda s: sim.simulate(s, start="known", tamper=tof_from(100, "no_target")), lambda e: holds(e, 99)),
    "tof_covered_from_20s": (lambda s: sim.simulate(s, start="known", tamper=tof_from(100, "covered")), lambda e: holds(e, 99)),
    "tof_stuck_from_20s": (lambda s: sim.simulate(s, start="known", tamper=tof_from(100, "stuck")), lambda e: holds(e, 99)),
    "wheel_slip_5s": (lambda s: sim.simulate(s, start="known", truth=TP[:SLIP[0]] + [TP[SLIP[0] - 1]] * (SLIP[1] - SLIP[0]) + TP[SLIP[0]:], tamper=slip),
                      lambda e: (lambda a, m: (a[0] and m <= 0.5, a[1] + f"; worst error during slip + 5 s {m:.2f} m (≤0.5)"))(
                          relock(e, SLIP[1] - 1, 5), max(float(np.max(x[SLIP[0] - 1:SLIP[1] + 24])) for x in e))),
    "heading_bias_10deg_per_m": (lambda s: sim.simulate(s, odo_bias=math.radians(10)), std),
    "odo_jump_2m": (lambda s: sim.simulate(s, start="known", tamper=at_frame(200, lambda o: (o[0] + 2.0, o[1], o[2]))), lambda e: relock(e, 199, 5)),
    "odo_nan_once": (lambda s: sim.simulate(s, start="known", tamper=at_frame(200, lambda o: (float("nan"), o[1], o[2]))), lambda e: relock(e, 199, 5)),
    # 6 room colour
    "colour_true": (lambda s: sim.simulate(s, room_hist=colour("true", s)), std),
    "colour_lights_changed": (lambda s: sim.simulate(s, room_hist=colour("lights", s)), std),
    "colour_says_wrong_room": (lambda s: sim.simulate(s, room_hist=colour("wrong", s)), std),
    "colour_identical_rooms": (lambda s: sim.simulate(s, house=same_colours(), room_hist=colour("true", s)), std),
    "colour_missing_2_rooms": (lambda s: sim.simulate(s, house=some_colours_missing(), world_house=H, room_hist=colour("true", s)), std),
}


@pytest.fixture(scope="session")
def results(request):
    wanted = [it.callspec.params["case"] for it in request.session.items if hasattr(it, "callspec") and "case" in it.callspec.params
              and it.module.__name__.endswith("test_routes")
              and (request.config.getoption("--slow") or not it.get_closest_marker("slow"))]
    keys = [(c, s) for c in wanted for s in SEEDS]
    print(f"\nroute cases: {len(wanted)} x {len(SEEDS)} seeds")
    out = sim.runs([lambda c=c, s=s: CASES[c][0](s) for c, s in keys])
    res = {}
    for (c, _), r in zip(keys, out):
        res.setdefault(c, []).append(r)
    return res


CORE = ["baseline", "kidnap_other_room", "kidnap_same_room", "twin_rooms_tv_cue", "fixed_fridge_moved_3m", "tv_shows_fridge",
        "stuck_phantom_fridge", "odo_nan_once"]  # the fast set; everything else runs with --slow


@pytest.mark.parametrize("case", [c if c in CORE else pytest.param(c, marks=pytest.mark.slow) for c in CASES])
def test_route(case, results):
    ok, detail = CASES[case][1](results[case])
    sim.report(case, ok, detail)
