"""The patrol of agent/mission.py on Pollen's duck-sim: the real robotd walks the duck with its trained policies in the
MuJoCo house, our brain steers. It steers by robotd's odometry, tofd's 8x8 ToF and object sightings from the twin's
stand-in detector at the true pose (agent/duck_recorder.py's truth.json, which also keeps the score).

python -m agent.walk --house out/houses/<name> [--goals kitchen,hallway,living_room] [--goal-s 240]
Out: out/walk/<house>/trace.jsonl (same fields as a mission trace, plus wall time and the command sent), summary.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from agent.duck_rpc import Duck
from agent.localize import ParticleFilter, _wrap
from agent.loop import route
from agent.mission import ARRIVE, ARRIVE_SURE, HZ, MARGIN, SQUEEZE, SURE, TURN, VX, goals
from agent.twin_sim import ROOT, TwinSim


def gait(v, w):
    """The brain's intent → what Pollen's walking policy does (measured in this flat): it doesn't step below ~0.2 m/s
    forward or ~1.2 rad/s on the spot, walks at ~40 % of the asked speed, turns on the spot at ~0.9 rad/s."""
    if v < 0:
        return -0.3, 0.0
    if v < 0.05:
        return (0.0, math.copysign(1.5, w)) if abs(w) > 0.1 else (0.0, 0.0)
    return max(0.2, 1.6 * v), w


def main():
    ap = argparse.ArgumentParser(prog="agent.walk")
    ap.add_argument("--house", default=str(ROOT / "out/current"))
    ap.add_argument("--goals", default="", help="room names in visiting order (default: all, as the mission does)")
    ap.add_argument("--goal-s", type=float, default=240)
    a = ap.parse_args()
    house_dir = Path(a.house).resolve()
    out = ROOT / "out/walk" / house_dir.name
    sim = TwinSim(house_dir)
    frame = json.loads((house_dir / "sim_frame.json").read_text())
    c, s, (px, py) = math.cos(frame["theta"]), math.sin(frame["theta"]), frame["pos"]

    def truth():  # MuJoCo world → house frame
        d = json.loads((out / "truth.json").read_text())
        wx, wy = d["xyz"][0] - px, d["xyz"][1] - py
        return (c * wx + s * wy, -s * wx + c * wy, float(_wrap(d["yaw"] - frame["theta"]))), d["xyz"][2]

    hmap, rooms = sim.world, [r["name"] for r in sim.house["rooms"]]
    room_name = lambda x, y: rooms[k] if (k := int(hmap.room(x, y))) >= 0 else ""  # noqa: E731
    todo = goals(sim)
    if a.goals:
        todo = [next(g for g in todo if g["name"] == n) for n in a.goals.split(",")]
    duck = Duck()
    while duck.state is None or duck.tof is None:
        time.sleep(0.1)
    pf = ParticleFilter.anywhere(hmap, seed=0)
    true, _ = truth()
    start, last, seq = true, duck.state["odom"], None
    t0 = time.time()
    trace, results, bumps, found, back, history = [], [], 0, None, 0, []
    think, steps, plans = 0.0, 0, 0  # the brain's own compute
    for g in todo:
        g0, path, last_plan, goal_bumps = time.time(), None, -9.0, 0
        while time.time() - g0 < a.goal_s and goal_bumps < 12:
            tick, t = time.time(), time.time() - t0
            true, trunk_z = truth()
            objs = sim.objects(*true)
            c0 = time.perf_counter()
            od = duck.state["odom"]  # odometry since the last tick, in the duck's own frame
            dx, dy = od["position"][0] - last["position"][0], od["position"][1] - last["position"][1]
            cl, sl = math.cos(last["yaw"]), math.sin(last["yaw"])
            pf.predict(dx * cl + dy * sl, -dx * sl + dy * cl, float(_wrap(od["yaw"] - last["yaw"])))
            last = od
            if duck.tof["seq"] != seq:
                seq = duck.tof["seq"]
                pf.update_tof(np.array(duck.tof["distance_mm"]), np.array(duck.tof["status"]))
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
                    results.append({"goal": g["name"], "t": round(time.time() - g0, 1), "declared": True,
                                    "true_dist": round(math.hypot(true[0] - g["xy"][0], true[1] - g["xy"][1]), 3),
                                    "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1])})
                    trace.append({"t": round(t, 2), "event": "arrive", "goal": g["name"], **results[-1]})
                    print("arrived", results[-1], flush=True)
                    break
                if path is None or t - last_plan > 2.0:
                    path, last_plan = route(hmap, est[:2], g["xy"], MARGIN) or route(hmap, est[:2], g["xy"], SQUEEZE), t
                    plans += 1
                mode = "go" if path else "stuck"
                if path:
                    ahead = next((p for p in path if math.hypot(p[0] - est[0], p[1] - est[1]) > 0.12), path[-1])
                    e = float(_wrap(math.atan2(ahead[1] - est[1], ahead[0] - est[0]) - est[2]))
                    w = max(-TURN, min(TURN, 2.0 * e))
                    v = VX * max(0.0, math.cos(e)) ** 3
                    if sim.clearance[int((est[1] - hmap.o[1]) / 0.02) + 1, int((est[0] - hmap.o[0]) / 0.02) + 1] < MARGIN + 0.02:
                        v = min(v, 0.08)
            think, steps = think + time.perf_counter() - c0, steps + 1
            vx, vyaw = gait(v, w)
            duck.move(vx, vyaw)
            history = (history + [(vx, true)])[-2 * HZ:]  # walking for 2 s and the body hasn't moved: a bump
            bumped = vx > 0 and len(history) == 2 * HZ and all(h[0] > 0 for h in history) and \
                math.hypot(true[0] - history[0][1][0], true[1] - history[0][1][1]) < 0.02
            if bumped:
                bumps += 1
                back, last_plan, goal_bumps, history = 5, -9.0, goal_bumps + 1, []
            trace.append({"t": round(t, 2), "wall": round(tick, 3), "true": [round(v_, 3) for v_ in true],
                          "est": [round(v_, 3) for v_ in est], "conf": round(conf, 2), "mode": mode,
                          "aim": [round(ahead[0], 2), round(ahead[1], 2)] if ahead else None, "cmd": [round(vx, 2), round(vyaw, 2)],
                          "goal": g["name"], "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1]),
                          "seen": [o["cls"] for o in objs], "bump": bumped, "trunk_z": round(trunk_z, 3)})
            time.sleep(max(0.0, 1 / HZ - (time.time() - tick)))
        else:
            results.append({"goal": g["name"], "t": round(time.time() - g0, 1), "declared": False,
                            "why": "stuck" if goal_bumps >= 12 else "time"})
            print("gave up", results[-1], flush=True)
    for _ in range(5):
        duck.move(0.0, 0.0)
        time.sleep(0.1)
    sure = [s for s in trace if "conf" in s and s["conf"] >= SURE]
    summary = {"house": house_dir.name, "start": [round(v, 3) for v in start], "goals": [g["name"] for g in todo],
               "goal_points": todo, "declared": sum(r["declared"] for r in results),
               "reached": sum(r["declared"] and r["true_dist"] < 0.5 for r in results),
               "found_s": round(found, 1) if found is not None else None, "time_s": round(time.time() - t0, 1), "bumps": bumps,
               "think_ms_per_step": round(1000 * think / max(1, steps), 2), "plans": plans,
               "room_right_pct": round(100 * np.mean([s["room_said"] == s["room_true"] for s in sure]), 1) if sure else None,
               "err_when_sure_median_m": round(float(np.median([math.hypot(s["est"][0] - s["true"][0], s["est"][1] - s["true"][1]) for s in sure])), 3) if sure else None,
               "fell": any(s["trunk_z"] < 0.08 for s in trace if "trunk_z" in s), "results": results}
    (out / "trace.jsonl").write_text("\n".join(json.dumps(s) for s in trace))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "goal_points"}, indent=1))


if __name__ == "__main__":
    main()
