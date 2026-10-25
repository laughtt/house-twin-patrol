"""A fast stand-in for eval/localize_test.py (no MuJoCo): the same route, odometry error model and simulated detector,
with the ToF ray-cast on the world's own occupancy grid. Every input can be tampered with per frame."""
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.localize import TOF_AZ, TOF_MAX, HouseMap, ParticleFilter  # noqa: E402

REAL = ROOT / "out/current/house.json"
HZ, SPEED, TURN = 5, 0.25, 0.8
TOF_FN = None  # set to another ToF simulator (test_mujoco.py) for every simulate() call
HFOV = 2 * math.atan(240 / (320 / math.tan(math.radians(57.7) / 2)))
ROUTE = [(3.3, 2.4), (3.45, 0.9), (2.0, 0.9), ("face", -math.pi / 2), (3.45, 0.95), (3.45, 2.3), (1.0, 2.3),
         ("face", math.pi), (3.45, 2.3), (3.45, 6.0), ("face", math.pi / 2), (1.0, 6.0)]


def real_house():
    if not REAL.exists():  # a fresh clone: the real flat's twin is private (out/ is not in the repo)
        import pytest
        pytest.skip("needs out/current/house.json, the real flat's twin (not in the repo)", allow_module_level=True)
    return json.loads(REAL.read_text())


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def truth_path(route=ROUTE):
    x, y = route[0]
    yaw = math.atan2(route[1][1] - y, route[1][0] - x)
    poses = [(x, y, yaw)]
    for step in route[1:]:
        goal = step[1] if step[0] == "face" else math.atan2(step[1] - y, step[0] - x)
        while abs(d := wrap(goal - yaw)) > 1e-3:
            yaw += max(-TURN / HZ, min(TURN / HZ, d))
            poses.append((x, y, yaw))
        if step[0] != "face":
            n = max(1, round(math.hypot(step[0] - x, step[1] - y) / (SPEED / HZ)))
            poses += [(x + (step[0] - x) * k / n, y + (step[1] - y) * k / n, yaw) for k in range(1, n + 1)]
            x, y = step
    return poses


def tof(world, x, y, yaw, rng):
    """8x8 frame like the microduck simulator: rows 3-4 level; status 5 valid, 255 nothing within 4 m."""
    d = world.raycast(np.full(8, x), np.full(8, y), yaw + TOF_AZ)
    dist, status = np.zeros((8, 8)), np.full((8, 8), 255)
    ok = d <= TOF_MAX
    meas = np.maximum(d + rng.normal(0, 0.003 + 0.02 * d / TOF_MAX), 0)
    dist[:, ok], status[:, ok] = meas[ok] * 1000, 5
    return dist.ravel(), status.ravel()


def detect(world, x, y, yaw, rng, hit=0.7, wrong=0.1, false=0.05, classes=None):
    """eval/localize_test.py seen_objects(): true landmarks in view and not hidden, with misses and mistakes."""
    classes = classes or sorted({lm["cls"] for lm in world.landmarks}) or ["chair"]
    out = []
    for lm in world.landmarks:
        r, rad = math.hypot(lm["center"][0] - x, lm["center"][1] - y), max(lm["size"][:2]) / 2
        b = wrap(math.atan2(lm["center"][1] - y, lm["center"][0] - x) - yaw)
        if r > 4.5 or abs(b) > HFOV / 2 or rng.random() > hit or \
                world.raycast(np.array([x]), np.array([y]), np.array([yaw + b]))[0] < min(r - rad - 0.15, 4):
            continue
        out.append({"cls": lm["cls"] if rng.random() > wrong else str(rng.choice(classes)), "bearing": b + rng.normal(0, math.radians(3)),
                    "range": max(r - 0.7 * rad, 0.2) * (1 + rng.normal(0, 0.1)) if rng.random() < 0.6 else None})
    if rng.random() < false:
        out.append({"cls": str(rng.choice(classes)), "bearing": float(rng.uniform(-HFOV / 2, HFOV / 2)), "range": None})
    return out


def odometry(fwd, turn, rng, bias=0.05):
    """4 % long, 2 % sideways drift, heading bias `bias` rad per metre (0.05 ≈ 3°/m), plus noise."""
    return (fwd * 1.04 + rng.normal(0, 0.02 * abs(fwd)), 0.02 * abs(fwd) + rng.normal(0, 0.03 * abs(fwd)),
            turn * 1.03 + bias * abs(fwd) + rng.normal(0, 0.01 * abs(turn) + 0.01 * abs(fwd)))


def simulate(seed=0, house=None, world_house=None, truth=None, start="unknown", use_tof=True, use_objects=True, n=2000,
             jumps=(), odo_bias=0.05, tamper=None, room_hist=None, det_kw=None, world_lms=None, tof_fn=None, ret_pf=False):
    """Run the filter along `truth` (default: the harness route). Returns per-frame position errors (m).

    jumps: frame indices where the duck is carried (truth jumps, odometry reports nothing).
    tamper(k, pose, odo, dets, tof) → (odo, dets, tof): adversarial edits per frame.
    room_hist(k, pose, world) → H-S histogram fed to update_room, or None.
    world_lms: landmarks as they really stand (default: the map's).
    tof_fn(x, y, yaw) → (dist, status): another ToF simulator (MuJoCo). ret_pf: also return the filter."""
    house = house or real_house()
    hmap, world = HouseMap(house), HouseMap(world_house or house)
    if world_lms is not None:
        world.landmarks = world_lms
    truth = truth or truth_path()
    rng = np.random.default_rng(7 + seed)
    pf = ParticleFilter.anywhere(hmap, n=n, seed=3 + seed) if start == "unknown" else ParticleFilter(hmap, *truth[0], n=n, seed=3 + seed)
    err = []
    for k in range(1, len(truth)):
        (ax, ay, ayaw), (bx, by, byaw) = truth[k - 1], truth[k]
        if k in jumps:
            odo = (0.0, 0.0, 0.0)
        else:
            odo = odometry((bx - ax) * math.cos(ayaw) + (by - ay) * math.sin(ayaw), wrap(byaw - ayaw), rng, odo_bias)
        dets = detect(world, bx, by, byaw, rng, **(det_kw or {})) if use_objects else []
        t = (tof_fn or TOF_FN)(bx, by, byaw) if (tof_fn or TOF_FN) else tof(world, bx, by, byaw, rng)
        if tamper:
            odo, dets, t = tamper(k, (bx, by, byaw), odo, dets, t)
        pf.predict(*odo)
        if use_tof:
            pf.update_tof(*t)
        if use_objects:
            pf.update_landmarks(dets, HFOV)
        if room_hist and (h := room_hist(k, (bx, by, byaw), world)) is not None:
            pf.update_room(h)
        e = pf.estimate()
        err.append(math.hypot(e[0] - bx, e[1] - by) if np.isfinite(e[:2]).all() else float("inf"))
    return (np.array(err), pf) if ret_pf else np.array(err)


def lock(err, start=0, tol=0.3, hold_s=2.0):
    """Seconds from frame `start` until the error stays under tol for hold_s (None: never)."""
    m = round(hold_s * HZ)
    ok = np.convolve(np.asarray(err[start:]) < tol, np.ones(m), "valid") == m
    return round(int(np.argmax(ok)) / HZ, 1) if ok.any() else None


def within(err, a=0, b=None, tol=0.3):
    """Share of frames a..b with the error under tol."""
    return float(np.mean(np.asarray(err[a:b]) < tol))


def after_lock_median(err, start=0):
    t = lock(err, start)
    return None if t is None else float(np.median(err[start + round(t * HZ):]))


REPORT = []  # (case, passed, numbers) for the summary table at the end of the pytest run
_TASKS = []


def report(name, ok, detail):
    REPORT.append((name, bool(ok), detail))
    assert ok, f"{name}: {detail}"


def runs(tasks, workers=None):
    """Zero-argument callables (lambdas are fine) run in parallel, forked so nothing needs pickling."""
    global _TASKS
    _TASKS = list(tasks)
    with ProcessPoolExecutor(min(workers or os.cpu_count(), len(_TASKS)), mp_context=multiprocessing.get_context("fork")) as ex:
        return list(ex.map(_run, range(len(_TASKS))))


def _run(i):
    return _TASKS[i]()
