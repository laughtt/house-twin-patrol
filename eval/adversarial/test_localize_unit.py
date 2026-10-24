"""Map edge cases and numerical robustness of agent/localize.py, driven directly (no route)."""
import copy
import math
import time

import numpy as np

import sim
from sim import HFOV, HouseMap, ParticleFilter, real_house

H = real_house()
POSE = (3.45, 1.45, math.pi / 2)  # kitchen walkway, facing north (on the harness route)


def tof_at(m, pose, seed=0):
    return sim.tof(m, *pose, np.random.default_rng(seed))


def near(pf, pose, tol=0.3):
    e = pf.estimate()
    return bool(np.isfinite(e).all() and math.hypot(e[0] - pose[0], e[1] - pose[1]) < tol), e


def tiny_house(**kw):
    h = {"walls": [{"a": [0, 0], "b": [3, 0], "t": 0.1}, {"a": [3, 0], "b": [3, 3], "t": 0.1},
                   {"a": [3, 3], "b": [0, 3], "t": 0.1}, {"a": [0, 3], "b": [0, 0], "t": 0.1}],
         "objects": [], "landmarks": [], "rooms": [{"name": "r", "rects": [[0.1, 0.1, 2.9, 2.9]]}]}
    h.update(kw)
    return h


# ---- 7 map edge cases ------------------------------------------------------------------------------------------------
def test_map_no_landmarks():
    m = HouseMap(tiny_house())
    pf = ParticleFilter.anywhere(m, n=500)
    pf.update_landmarks([{"cls": "refrigerator", "bearing": 0.1, "range": 1.0}], HFOV)
    sim.report("map_no_landmarks", np.isfinite(pf.estimate()).all(), "update_landmarks with an empty landmark list runs")


def test_map_bad_landmark_entries():
    lms = [{"cls": "", "center": [1, 1], "size": [1, 1, 1], "weight": 1}, {"center": [1, 1], "size": [1, 1, 1], "weight": 1},
           {"cls": "tv", "center": [1, 1], "size": [1, 1, 1], "weight": 0}, {"cls": "tv", "center": [1, 1], "size": [1, 1, 1]},
           {"cls": "tv", "center": [1, 1], "size": [1, 1, 1], "weight": 0.6}]
    m = HouseMap(tiny_house(landmarks=lms))
    sim.report("map_landmarks_without_cls_or_weight", len(m.landmarks) == 1, f"kept {len(m.landmarks)} of 5 (need 1: the only one with cls and weight > 0)")


def test_map_no_rooms():
    m = HouseMap(tiny_house(rooms=[]))
    pf = ParticleFilter.anywhere(m, n=500)
    pf.update_room(np.full(128, 1 / 128))
    sim.report("map_rooms_empty", np.isfinite(pf.p).all() and pf.p[:, 0].max() > 2, "anywhere() covers the free floor, update_room is a no-op")


def test_map_room_with_no_free_floor():
    """A room whose only rectangle lies inside a wall (a tiny room the floor plan got wrong)."""
    m = HouseMap(tiny_house(rooms=[{"name": "closet", "rects": [[2.97, 1.0, 3.0, 1.03]]}]))
    try:
        pf = ParticleFilter.anywhere(m, n=500)
        ok, why = np.isfinite(pf.p).all(), "anywhere() fell back to some floor"
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"anywhere() raised {type(e).__name__}: {e}"
    sim.report("map_tiny_room_no_free_cell", ok, why)


def test_map_negative_coordinates():
    """house.json's origin is the lower-left of the LARGEST space; a smaller room west or south of it has x or y < 0.
    Shift the real house by (-1.5, -1.5): the map must still know that the duck's spot is floor and see the same walls."""
    h, s = copy.deepcopy(H), 1.5
    for w in h["walls"]:
        w["a"], w["b"] = [w["a"][0] - s, w["a"][1] - s], [w["b"][0] - s, w["b"][1] - s]
    for o in h["objects"] + h["landmarks"]:
        o["center"] = [o["center"][0] - s, o["center"][1] - s]
    for r in h["rooms"]:
        r["rects"] = [[x0 - s, y0 - s, x1 - s, y1 - s] for x0, y0, x1, y1 in r["rects"]]
    a, b = HouseMap(H), HouseMap(h)
    x, y, yaw = 1.0, 2.3, math.pi  # on the route, facing the west wall
    da = a.raycast(np.full(8, x), np.full(8, y), yaw + sim.TOF_AZ)
    db = b.raycast(np.full(8, x - s), np.full(8, y - s), yaw + sim.TOF_AZ)
    ok = bool(b.free(x - s, y - s)) and np.allclose(da, db, atol=0.05)
    sim.report("map_negative_coordinates", ok, f"free at shifted pose: {bool(b.free(x - s, y - s))}; ToF ranges {da.round(2).tolist()} vs shifted {db.round(2).tolist()}")


def test_real_landmarks_inside_walls():
    xs = [p for w in H["walls"] for p in (w["a"][0], w["b"][0])]
    ys = [p for w in H["walls"] for p in (w["a"][1], w["b"][1])]
    out = [(lm["label"], lm["center"]) for lm in H["landmarks"] if not (min(xs) <= lm["center"][0] <= max(xs) and min(ys) <= lm["center"][1] <= max(ys))]
    sim.report("real_map_landmarks_inside_walls", not out, f"outside the walls' box [{min(xs)}..{max(xs)}] x [{min(ys)}..{max(ys)}]: {out}")


# ---- 8 numerical robustness ------------------------------------------------------------------------------------------
def _poisoned(det=None, odo=None, tof=None):
    m = HouseMap(H)
    pf = ParticleFilter(m, *POSE, n=2000)
    for k in range(6):
        pf.predict(*(odo if odo and k == 2 else (0.0, 0.0, 0.0)))
        d, s = tof_at(m, POSE, k)
        if tof and k == 2:
            d = d.copy()
            d[3 * 8 + 4] = tof
        pf.update_tof(d, s)
        pf.update_landmarks([det] if det else [], HFOV)
    return pf


def test_nan_bearing():
    pf = _poisoned(det={"cls": "refrigerator", "bearing": float("nan"), "range": None})
    ok, e = near(pf, POSE)
    sim.report("nan_bearing_x6", ok and np.isfinite(pf.p).all(), f"finite particles: {np.isfinite(pf.p).all()}, estimate {np.round(e, 2).tolist()}")


def test_nan_range():
    pf = _poisoned(det={"cls": "refrigerator", "bearing": 0.2, "range": float("nan")})
    ok, e = near(pf, POSE)
    sim.report("nan_range_x6", ok and np.isfinite(pf.p).all(), f"finite particles: {np.isfinite(pf.p).all()}, estimate {np.round(e, 2).tolist()}")


def test_inf_range():
    pf = _poisoned(det={"cls": "refrigerator", "bearing": 0.2, "range": float("inf")})
    ok, e = near(pf, POSE)
    sim.report("inf_range_x6", ok and np.isfinite(pf.p).all(), f"finite particles: {np.isfinite(pf.p).all()}, estimate {np.round(e, 2).tolist()}")


def test_negative_range():
    pf = _poisoned(det={"cls": "refrigerator", "bearing": 0.2, "range": -2.0})
    ok, e = near(pf, POSE)
    sim.report("negative_range_x6", ok, f"estimate {np.round(e, 2).tolist()}")


def test_nan_odometry():
    pf = _poisoned(odo=(float("nan"), 0.0, 0.0))
    ok, e = near(pf, POSE)
    sim.report("nan_odometry_once", ok, f"finite particles: {int(np.isfinite(pf.p).all(1).sum())}/2000, estimate {np.round(e, 2).tolist()}")


def test_nan_tof_zone_keeps_evidence():
    """Half the guesses at the duck, half in the living room. One ToF zone reads NaN (a driver bug): the other 15 zones
    should still say where the duck is, not reset every weight to uniform."""
    m = HouseMap(H)
    pf = ParticleFilter(m, *POSE, n=2000)
    pf.p[1000:] = [2.0, 5.5, 0.0]
    d, s = tof_at(m, POSE)
    d = d.copy()
    d[3 * 8 + 4] = np.nan
    pf.update_tof(d, s)
    share = float(pf.w[np.hypot(pf.p[:, 0] - POSE[0], pf.p[:, 1] - POSE[1]) < 0.5].sum())  # by place: resampling reorders particles
    sim.report("nan_tof_zone", share > 0.9, f"weight on the true place after the update {share:.2f} (>0.9; 0.50 = evidence thrown away)")


def test_all_zero_weights():
    pf = ParticleFilter(HouseMap(H), *POSE, n=500)
    pf.w[:] = 0
    pf._normalise()
    sim.report("all_zero_weights", np.allclose(pf.w, 1 / 500), "normalise falls back to uniform")


def test_tiny_particle_counts():
    errs = []
    for n in (1, 2, 10):
        try:
            m = HouseMap(H)
            pf = ParticleFilter.anywhere(m, n=n)
            for _ in range(5):
                pf.predict(0.05, 0, 0)
                pf.update_tof(*tof_at(m, POSE))
                pf.update_landmarks([{"cls": "refrigerator", "bearing": 0.0, "range": 1.0}] * 4, HFOV)
                pf.estimate()
        except Exception as e:  # noqa: BLE001
            errs.append(f"n={n}: {type(e).__name__}: {e}")
    sim.report("n_1_2_10_particles", not errs, errs or "runs")


def test_many_detections_time():
    """A cluttered frame: 30 detections of a fixed class (5 counters on the map) on 2000 guesses, once per 5 Hz frame."""
    m = HouseMap(H)
    pf = ParticleFilter.anywhere(m)
    dets = [{"cls": "sink", "bearing": b, "range": 1.5} for b in np.linspace(-0.3, 0.3, 30)]
    t = time.perf_counter()
    pf.update_landmarks(dets, HFOV)
    dt = time.perf_counter() - t
    t = time.perf_counter()
    pf.update_tof(*tof_at(m, POSE))
    dtof = time.perf_counter() - t
    sim.report("30_detections_per_frame_time", dt <= 0.2, f"update_landmarks {dt * 1000:.0f} ms (≤200 ms, one 5 Hz frame); update_tof {dtof * 1000:.0f} ms")


def test_estimate_spread_true_cluster_vs_tight_wrong_blob():
    """2/3 of the guesses spread around the duck (σ 12 cm, centred on a 30 cm cell corner), 1/3 in a tight blob 3 m away.
    No previous estimate. The answer must be the duck's place: the fixed 30 cm grid splits the true cluster four ways."""
    m = HouseMap(H)
    pf = ParticleFilter(m, 1.5, 1.5, 0.0, n=3000)
    g = np.random.default_rng(0)
    pf.p[:2000, :2] = g.normal([1.5, 1.5], 0.12, (2000, 2))
    pf.p[2000:, :2] = g.normal([2.0, 4.5], 0.03, (1000, 2))
    pf.w[:] = 1 / 3000
    e = pf.estimate()
    d = math.hypot(e[0] - 1.5, e[1] - 1.5)
    sim.report("estimate_grid_split", d < 0.3, f"estimate {np.round(e[:2], 2).tolist()} is {d:.2f} m from the 2/3 cluster at (1.5, 1.5)")


def test_estimate_switches_when_evidence_says_so():
    """Half the guesses at the truth, half in the living room, and the last estimate was the living room (hysteresis).
    True ToF readings arrive: the estimate must move to the truth within 5 frames (1 s)."""
    m = HouseMap(H)
    pf = ParticleFilter(m, *POSE, n=2000, spread=(0.03, 0.03))
    pf.p[1000:] = np.c_[np.random.default_rng(1).normal([2.0, 5.5], 0.03, (1000, 2)), np.zeros(1000)]
    pf.last = (2.0, 5.5)
    frames = None
    for k in range(10):
        pf.predict(0, 0, 0)
        pf.update_tof(*tof_at(m, POSE, k))
        if near(pf, POSE)[0]:
            frames = k + 1
            break
    sim.report("estimate_hysteresis_switch", frames is not None and frames <= 5, f"moved to the truth after {frames} frames (≤5)")


def test_bearing_outside_pi():
    m = HouseMap(H)
    a, b = ParticleFilter(m, *POSE, n=500, seed=1), ParticleFilter(m, *POSE, n=500, seed=1)
    a.update_landmarks([{"cls": "refrigerator", "bearing": 0.3, "range": None}], HFOV)
    b.update_landmarks([{"cls": "refrigerator", "bearing": 0.3 + 4 * math.pi, "range": None}], HFOV)
    sim.report("bearing_plus_4pi", np.allclose(a.w, b.w), "bearing + 4π scores like the bearing")
