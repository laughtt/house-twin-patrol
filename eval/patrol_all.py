"""The duck's patrol in every built house we have: the real flats, the ARKitScenes scans and the generated homes.

uv run --python 3.12 --with mujoco --with opencv-python-headless==5.0.0.93 python eval/patrol_all.py [--seeds 4]
Out: out/mission/all.json (one row per house) and a printed table.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.mission import episode, goals  # noqa: E402
from agent.twin_sim import TwinSim  # noqa: E402

HOUSES = [ROOT / "out/houses/lidar-20261023-172353", ROOT / "out/houses/lidar-20261024-120616",
          *sorted(p.parent for p in (ROOT / "out/arkitscenes").glob("*/house/house.json")),
          *sorted(p.parent for p in (ROOT / "out/synth_houses").glob("h*/house.json"))]


def name(h):
    return h.parent.name if h.name == "house" else h.name


if __name__ == "__main__":
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 4
    jobs = [(h, s, ROOT / "out/mission" / name(h)) for h in HOUSES if goals(TwinSim(h)) for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        eps = list(pool.map(episode, jobs))
    rows = []
    for h in HOUSES:
        E = [e for (hh, _, _), e in zip(jobs, eps) if hh == h]
        if not E:
            rows.append({"house": name(h), "note": "no goal the duck can stand at"})
            continue
        found = [e["found_s"] for e in E if e["found_s"] is not None]
        sure = [e["err_when_sure_median_m"] for e in E if e["err_when_sure_median_m"] is not None]
        rooms = [e["room_right_pct"] for e in E if e["room_right_pct"] is not None]
        rows.append({"house": name(h), "goals": E[0]["goals"], "reached": f'{sum(e["reached"] for e in E)}/{sum(e["goals"] for e in E)}',
                     "found_s": round(float(np.median(found)), 1) if found else None, "found_all": f"{len(found)}/{len(E)}",
                     "err_cm": round(100 * float(np.median(sure)), 1) if sure else None,
                     "room_right_pct": round(float(np.median(rooms)), 1) if rooms else None, "bumps": sum(e["bumps"] for e in E),
                     "think_ms": round(float(np.mean([e["think_ms_per_step"] for e in E])), 2), "plans": sum(e["plans"] for e in E)})
    (ROOT / "out/mission" / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "all.json")).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))
