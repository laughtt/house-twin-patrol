"""Self-location sandbox: the patrol (agent.mission.episode) with the filter before spin match ("base") and with each
idea in .claude/2026-10-25-selfloc-sandbox.md, on the same houses, seeds and sensor.

uv run --python 3.12 --with mujoco --with opencv-python-headless==5.0.0.93 python eval/selfloc.py \
    [--variants base,fit,spin+rowfix+range] [--sensors sim,real] [--houses default|all|<dir>,<dir>] [--seeds 8] [--run name]

A variant is '+'-joined settings of agent/localize.py's ParticleFilter ("base" = none: the filter before spin match):
  rowfix  a zone is read from the rows that measured it (one bad row no longer turns 1 m into ~3 m)
  range   the filter's range (TOF_MAX) = the sensor's
  fit     confidence capped at 0.5 unless the measured zones fit the map (≥ 60 % within 15 cm over 5 s)
  spin    not sure until one full turn of ToF matches one place clearly; then the guesses are re-seeded on the matches
  relook  with spin: fit under 50 % for 3 s → stack a turn and match again
  jump    with spin+fit: an estimate > 1 m + 10 % of the walk from the last trusted pose is a false positive: keep the
          guesses near that pose; none left → match again, nearby first
Sensors: sim (4 m, every zone), real (a stand-in: 2.5 m, 20 % of zones dropped).
Out: out/selfloc/<run>/results.json and report.html, and a printed table.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent.localize as L  # noqa: E402
from agent.mission import episode  # noqa: E402

SENSORS = {"sim": (4.0, 0.0), "real": (2.5, 0.2)}
ALL = "spin+rowfix+range+fit+relook+jump"  # the duck's default
DEFAULT = ["base", "rowfix+range", "rowfix+range+fit", "spin+rowfix+range", "spin+rowfix+range+fit+relook", ALL]
HOUSES = ["out/houses/lidar-20261024-120616", "out/houses/lidar-20261023-172353", "out/houses/synth-multi"]


def variant(flags, cap):
    """The real filter (agent/localize.py) with these settings; no flag = the filter before spin match."""
    settings = {k: k in flags for k in ("rowfix", "spin", "fit", "relook", "jump")}
    return SimpleNamespace(anywhere=partial(L.ParticleFilter.anywhere, **settings, tof_max=cap if "range" in flags else L.TOF_MAX))


def run(job):
    house, sensor, var, seed = job
    cap, drop = SENSORS[sensor]

    def sense(sim, dist, status):
        d, s = np.array(dist, float), np.array(status)
        s[(d / 1000 > cap) | (sim.rng.random(64) < drop)] = 255
        return d, s
    t = time.time()
    e = episode((ROOT / house, seed, None), pf_cls=variant(set(var.split("+")), cap), sense=sense if sensor != "sim" else None)
    return {**{k: v for k, v in e.items() if k != "results"}, "house": Path(house).name, "sensor": sensor, "variant": var, "wall_s": time.time() - t}


def table(eps):
    rows = []
    for key in dict.fromkeys((e["house"], e["sensor"], e["variant"]) for e in eps):
        E = [e for e in eps if (e["house"], e["sensor"], e["variant"]) == key]
        found = [e["found_s"] for e in E if e["found_s"] is not None]
        fc = [e["false_conf_pct"] for e in E if e["false_conf_pct"] is not None]
        rows.append({"house": key[0], "sensor": key[1], "variant": key[2], "seeds": len(E),
                     "false_conf_pct": round(float(np.mean(fc)), 1) if fc else None,
                     "wrong_lock_seeds": sum(e["first_wrong_s"] is not None for e in E),
                     "false_arrivals": sum(e["false_arrivals"] for e in E),
                     "found": len(found), "found_s": round(float(np.median(found)), 1) if found else None,
                     "reached": sum(e["reached"] for e in E), "goals": sum(e["goals"] for e in E),
                     "room_right_pct": round(float(np.median([e["room_right_pct"] for e in E if e["room_right_pct"] is not None] or [0])), 1),
                     "bumps": sum(e["bumps"] for e in E), "think_ms": round(float(np.mean([e["think_ms_per_step"] for e in E])), 1)})
    return rows


COLS = [("false_conf_pct", "false conf %", -1), ("wrong_lock_seeds", "wrong-lock seeds", -1), ("false_arrivals", "false arrivals", -1),
        ("found", "found", 1), ("found_s", "found s", -1), ("reached", "reached", 1), ("room_right_pct", "room right %", 1),
        ("bumps", "bumps", -1), ("think_ms", "think ms", -1)]


def report(rows, path):
    base = {(r["house"], r["sensor"]): r for r in rows if r["variant"] == "base"}

    def cell(r, k, better):
        b, v = base.get((r["house"], r["sensor"]), {}).get(k), r[k]
        col = "" if b is None or v is None or v == b or r["variant"] == "base" else ("good" if (v - b) * better > 0 else "bad")
        txt = f'{v}/{r["goals"]}' if k == "reached" else f'{v}/{r["seeds"]}' if k in ("found", "wrong_lock_seeds") else v
        return f'<td class="{col}">{"—" if v is None else txt}</td>'
    body = "".join(f'<tr><td>{r["house"]}</td><td>{r["sensor"]}</td><td><code>{r["variant"]}</code></td>'
                   + "".join(cell(r, k, s) for k, _, s in COLS) + "</tr>" for r in rows)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-location Sandbox</title><style>
:root{{--bg:#f7f7f5;--ink:#1d1d1f;--line:#ddd;--good:#d6f5df;--bad:#fbdcdc}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151517;--ink:#eee;--line:#333;--good:#1d4a2c;--bad:#5a2323}}}}
body{{margin:0;padding:20px 16px;background:var(--bg);color:var(--ink);font:14px -apple-system,system-ui,sans-serif}}
.t{{overflow-x:auto}}table{{border-collapse:collapse}}th,td{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
.good{{background:var(--good)}}.bad{{background:var(--bad)}}</style></head><body>
<h1>Self-location sandbox</h1><p>Green / red: better / worse than <code>base</code> (today's filter) on the same house, sensor and seeds.
Trust false conf, wrong-lock seeds and false arrivals most; reached is mostly navigation.</p>
<div class="t"><table><tr><th>house</th><th>sensor</th><th>variant</th>{"".join(f"<th>{h}</th>" for _, h, _ in COLS)}</tr>{body}</table></div></body></html>""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="eval/selfloc.py")
    ap.add_argument("--variants", default=",".join(DEFAULT))
    ap.add_argument("--sensors", default="sim,real")
    ap.add_argument("--houses", default="default")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--run", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()
    if a.houses == "all":
        from eval.patrol_all import HOUSES as ALL
        houses = [str(h.relative_to(ROOT)) for h in ALL]
    else:
        houses = HOUSES if a.houses == "default" else a.houses.split(",")
    jobs = [(h, s, v, k) for h in houses for s in a.sensors.split(",") for v in a.variants.split(",") for k in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        eps = list(pool.map(run, jobs))
    rows = table(eps)
    out = ROOT / "out/selfloc" / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"rows": rows, "episodes": eps}, indent=1))
    report(rows, out / "report.html")
    print(f"{'house':24}{'sensor':7}{'variant':32}" + "".join(f"{h:>17}" for _, h, _ in COLS))
    for r in rows:
        print(f'{r["house"]:24}{r["sensor"]:7}{r["variant"]:32}' + "".join(f'{str(r[k]):>17}' for k, _, _ in COLS))
    print(out / "report.html")
