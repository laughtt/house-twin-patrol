"""The duck's patrol, closed loop in its twin home: it is set down somewhere it doesn't know, finds itself (no stickers),
then visits every room (or, in a one-room scan, every landmark that stays put), steering by where it THINKS it is.
The twin knows the truth and keeps the score.

uv run --python 3.12 --with mujoco --with opencv-python-headless==5.0.0.93 python -m agent.mission [--seeds 8] [--house out/current]

Each step (5 Hz): sense (ToF + objects) → update the guesses → unsure? turn on the spot to look around : plan an A*
route from the guess and steer along it → at the goal, say which room it believes it is in.
Out: out/mission/<house>/seed<N>/trace.jsonl, summary.json per episode, and summary.json over all episodes.
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from agent.localize import ParticleFilter, _wrap
from agent.loop import follow, route
from agent.twin_sim import ROOT, TwinSim

HZ, VX, TURN = 5, 0.25, 0.8
SURE, ARRIVE, GOAL_S = 0.6, 0.25, 120  # confidence to act on the guess; how close is "there"; time per goal
ARRIVE_SURE = 0.85  # saying "I'm here" needs more: at 0.6 it sometimes claimed a goal it was 50 cm+ away from
MARGIN = 0.18  # route clearance: the duck's 10 cm half width + 8 cm for where it thinks it is vs where it is
SQUEEZE = 0.12  # when no route has that much room (a narrow door): 2 cm to spare, walked slowly


def goals(sim):
    """Room centres (of the room's largest rectangle: an L's arms can meet through a gap the duck can't pass) when the scan
    has several rooms, else the landmarks that stay put; each moved to the nearest spot the duck can stand on (≥ 30 cm
    from the thing it is named after)."""
    rooms = sim.house["rooms"]
    if len(rooms) > 1:
        big = lambda r: max(r["rects"], key=lambda a: (a[2] - a[0]) * (a[3] - a[1]))  # noqa: E731
        want = [(r["name"], np.array([(big(r)[0] + big(r)[2]) / 2, (big(r)[1] + big(r)[3]) / 2])) for r in rooms]
    else:
        lms = [lm for lm in sim.house["landmarks"] if lm.get("mobility") in ("fixed", "rare")]
        want = [(lm["label"], np.array(lm["center"])) for lm in lms]
    out, seen = [], set()
    for name, c in want:
        best = None
        for r in np.arange(0.3, 2.0, 0.05):  # the nearest free spot on rings around the centre
            for a in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                x, y = c[0] + r * math.cos(a), c[1] + r * math.sin(a)
                if sim.free(x, y, clear=MARGIN + 0.04) and sim.world.room(x, y) >= 0:  # room to stand, wider than the route margin
                    best = (x, y)
                    break
            if best:
                break
        if best and name not in seen:
            seen.add(name)
            out.append({"name": name, "xy": [round(best[0], 3), round(best[1], 3)]})
    return out


def episode(args, pf_cls=ParticleFilter, sense=None):
    house_dir, seed, out = args
    sim = TwinSim(house_dir, seed=seed)
    rng = np.random.default_rng(seed)
    hmap, rooms = sim.world, [r["name"] for r in sim.house["rooms"]]
    room_name = lambda x, y: rooms[k] if (k := int(hmap.room(x, y))) >= 0 else ""  # noqa: E731
    todo = goals(sim)
    k = int(rng.integers(len(todo))) if todo else 0
    todo = todo[k:] + todo[:k]  # start the round at a random goal
    true = start = sim.random_pose()
    pf = pf_cls.anywhere(hmap, seed=seed)
    t, trace, results, bumps = 0.0, [], [], 0
    found = None  # first moment it was sure and right
    back = 0  # steps left of backing off after a bump
    think, plans = 0.0, 0  # the brain's own compute (not the twin's), and how many routes it planned
    for g in todo:
        t0, path, last_plan, goal_bumps = t, None, -9.0, 0
        while t - t0 < GOAL_S and goal_bumps < 12:  # a patrol gives up on a spot it can't reach and moves on
            dist, status = sim.tof(*true)
            if sense:
                dist, status = sense(sim, dist, status)
            objs = sim.objects(*true)
            tick = time.perf_counter()
            pf.update_tof(dist, status)
            pf.update_landmarks(objs, sim.hfov)
            est, conf = pf.estimate(), pf.confidence()
            err = math.hypot(est[0] - true[0], est[1] - true[1])
            if found is None and conf >= SURE and err < 0.3:
                found = t
            mode, v, w, ahead = "look", 0.0, 0.6, None
            if back:
                mode, v, w, back = "back off", -0.1, 0.0, back - 1
            elif conf >= SURE:
                if math.hypot(est[0] - g["xy"][0], est[1] - g["xy"][1]) < ARRIVE and conf >= ARRIVE_SURE:
                    results.append({"goal": g["name"], "t": round(t - t0, 1), "declared": True,
                                    "true_dist": round(math.hypot(true[0] - g["xy"][0], true[1] - g["xy"][1]), 3),
                                    "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1])})
                    trace.append({"t": round(t, 1), "event": "arrive", "goal": g["name"], **results[-1]})
                    break
                ahead, reached = follow(path, est) if path else (None, False)
                if reached or t - last_plan > 2.0:  # a corner reached (plan again from where it really is), or now and then
                    path, last_plan = route(hmap, est[:2], g["xy"], MARGIN) or route(hmap, est[:2], g["xy"], SQUEEZE), t
                    plans += 1
                mode = "go" if path else "stuck"
                if path:  # steer at the next corner of the route until 12 cm from it (aiming past a corner cuts it)
                    if last_plan == t:  # just planned
                        ahead = follow(path, est)[0]
                    e = float(_wrap(math.atan2(ahead[1] - est[1], ahead[0] - est[0]) - est[2]))
                    w = max(-TURN, min(TURN, 2.0 * e))
                    v = VX * max(0.0, math.cos(e)) ** 3  # slow down in turns instead of cutting the corner
                    if sim.clearance[int((est[1] - hmap.o[1]) / 0.02) + 1, int((est[0] - hmap.o[0]) / 0.02) + 1] < MARGIN + 0.02:
                        v = min(v, 0.08)  # tight spot (a door): creep
            think += time.perf_counter() - tick
            true, odo, bumped = sim.move(true, v, w, 1 / HZ)
            bumps += bumped
            if bumped and v > 0:
                back, last_plan, goal_bumps = 3, -9.0, goal_bumps + 1  # back off 6 cm and plan again
            tick = time.perf_counter()
            pf.predict(*odo)
            think += time.perf_counter() - tick
            trace.append({"t": round(t, 1), "true": [round(true[0], 3), round(true[1], 3), round(true[2], 3)],
                          "est": [round(est[0], 3), round(est[1], 3), round(est[2], 3)], "conf": round(conf, 2), "mode": mode,
                          "aim": [round(ahead[0], 2), round(ahead[1], 2)] if ahead else None,
                          "goal": g["name"], "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1]),
                          "seen": [o["cls"] for o in objs], "bump": bool(bumped)})
            t += 1 / HZ
        else:
            results.append({"goal": g["name"], "t": round(t - t0, 1), "declared": False, "why": "stuck" if goal_bumps >= 12 else "time"})
    sure = [s for s in trace if "conf" in s and s["conf"] >= SURE]
    off = [math.hypot(s["est"][0] - s["true"][0], s["est"][1] - s["true"][1]) for s in sure]
    summary = {"seed": seed, "start": [round(v, 3) for v in start[:2]], "goals": len(todo),
               "declared": sum(r["declared"] for r in results),
               "reached": sum(r["declared"] and r["true_dist"] < 0.5 for r in results),
               "found_s": round(found, 1) if found is not None else None, "time_s": round(t, 1), "bumps": bumps,
               "think_ms_per_step": round(1000 * think / max(1, round(t * HZ)), 2), "plans": plans,
               "room_right_pct": round(100 * np.mean([s["room_said"] == s["room_true"] for s in sure]), 1) if sure else None,
               "err_when_sure_median_m": round(float(np.median(off)), 3) if sure else None,
               "false_conf_pct": round(100 * float(np.mean(np.array(off) > 0.5)), 1) if sure else None,  # sure but > 50 cm off
               "first_wrong_s": next((s["t"] for s, e in zip(sure, off) if e > 0.5), None),
               "false_arrivals": sum(r["declared"] and r["true_dist"] >= 0.5 for r in results), "results": results}
    if out:
        d = Path(out) / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "trace.jsonl").write_text("\n".join(json.dumps(s) for s in trace))
        (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(prog="agent.mission")
    ap.add_argument("--house", default=str(ROOT / "out/current"))
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    house_dir = Path(a.house).resolve()
    out = ROOT / "out/mission" / (house_dir.name if house_dir.name != "house" else house_dir.parent.name)
    with ProcessPoolExecutor(max_workers=8) as pool:
        eps = list(pool.map(episode, [(house_dir, s, out) for s in range(a.seeds)]))
    goal_list = goals(TwinSim(house_dir))
    total = {"house": house_dir.name, "episodes": len(eps), "goals": [g["name"] for g in goal_list],
             "goals_reached": f'{sum(e["reached"] for e in eps)}/{sum(e["goals"] for e in eps)}',
             "found_s_median": float(np.median([e["found_s"] for e in eps if e["found_s"] is not None])) if any(e["found_s"] is not None for e in eps) else None,
             "room_right_pct_median": float(np.median([e["room_right_pct"] for e in eps if e["room_right_pct"] is not None])),
             "err_when_sure_median_m": float(np.median([e["err_when_sure_median_m"] for e in eps if e["err_when_sure_median_m"] is not None])),
             "bumps_total": sum(e["bumps"] for e in eps), "per_episode": [{k: e[k] for k in ("seed", "reached", "goals", "found_s", "time_s", "bumps", "room_right_pct")} for e in eps]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({**total, "goal_points": goal_list}, indent=1))
    # one compact file for the replay page: per step [t, true x y yaw, guess x y yaw, confidence, mode, goal, aim x y, bump, seen]
    modes, names = ["look", "go", "back off", "stuck"], [g["name"] for g in goal_list]
    h = json.loads((house_dir / "house.json").read_text())  # the exact house the patrols ran in
    replay = {"summary": total, "goals": goal_list, "modes": modes, "episodes": [],
              "house": {k: h[k] for k in ("walls", "rooms", "objects", "landmarks")}}
    for e in eps:
        tr = [json.loads(s) for s in (out / f'seed{e["seed"]}' / "trace.jsonl").read_text().splitlines()]
        steps = [[s["t"], *s["true"], *s["est"], s["conf"], modes.index(s["mode"]), names.index(s["goal"]) if s["goal"] in names else -1,
                  *(s["aim"] or [None, None]), int(s["bump"]), "+".join(s["seen"])] for s in tr if "mode" in s]
        replay["episodes"].append({**{k: v for k, v in e.items()}, "steps": steps, "arrivals": [s for s in tr if s.get("event") == "arrive"]})
    (out / "mission.json").write_text(json.dumps(replay, separators=(",", ":")))
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    sys.exit(main())
