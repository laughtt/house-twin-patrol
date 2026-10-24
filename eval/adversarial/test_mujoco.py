"""Slow cross-check: the key route cases again with the ToF from MuJoCo (the microduck simulator on the real twin, as in
eval/localize_test.py) instead of sim.py's ray-cast on the map grid. Needs `--with mujoco` and `--slow`.
Uses the twin_eye_run.xml scene that eval/localize_test.py writes (run it once first)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

import sim

pytestmark = pytest.mark.slow
ROBOT = Path.home() / "Pollen/microduck_rl/src/mjlab_microduck/robot/microduck"
CHECK = ["baseline", "kidnap_other_room", "kidnap_same_room", "tv_shows_fridge", "fixed_fridge_moved_3m", "lights_off_unknown_start",
         "odo_nan_once"]
_mj = {}


def mujoco_tof(x, y, yaw):
    """Built lazily in each worker process, so forked workers don't share MuJoCo state."""
    if not _mj:
        import mujoco
        spec = importlib.util.spec_from_file_location("tof", ROBOT.parents[1] / "sim/tof.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mujoco.MjModel.from_xml_path(str(ROBOT / "twin_eye_run.xml"))
        frame = json.loads((sim.ROOT / "out/current/sim_frame.json").read_text())
        _mj.update(mujoco=mujoco, model=model, data=mujoco.MjData(model), tof=mod.Tof(model, model.site("tof").id, seed=1),
                   th=frame["theta"], pos=frame["pos"])
    m, th, (px, py) = _mj, _mj["th"], _mj["pos"]
    m["data"].mocap_pos[0] = [math.cos(th) * x - math.sin(th) * y + px, math.sin(th) * x + math.cos(th) * y + py, 0.2]
    m["data"].mocap_quat[0] = [math.cos((yaw + th) / 2), 0, 0, math.sin((yaw + th) / 2)]
    m["mujoco"].mj_forward(m["model"], m["data"])
    return m["tof"].frame(m["data"])


@pytest.fixture(scope="module")
def mj_results():
    if not (ROBOT / "twin_eye_run.xml").exists():
        pytest.skip("run eval/localize_test.py once: it writes the MuJoCo scene")
    from test_routes import CASES, SEEDS
    sim.TOF_FN = mujoco_tof
    keys = [(c, s) for c in CHECK for s in SEEDS]
    out = sim.runs([lambda c=c, s=s: CASES[c][0](s) for c, s in keys])
    sim.TOF_FN = None
    res = {}
    for (c, _), r in zip(keys, out):
        res.setdefault(c, []).append(r)
    return res, CASES


@pytest.mark.parametrize("case", CHECK)
def test_mujoco_route(case, mj_results):
    res, cases = mj_results
    ok, detail = cases[case][1](res[case])
    sim.report(f"mujoco_{case}", ok, detail)
