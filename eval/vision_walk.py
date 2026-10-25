"""The patrol with the duck's camera on, in the offline twin: each step the twin renders what the duck sees and OpenCV 5
reads it (room colour, the house's ArUco markers) into the filter that the route is planned from. Same decisions as
agent/walk.py (vision_decision events, the lost-for-LOST_S question) without Pollen's duck-sim. Compares ToF alone with
ToF + camera from the same starts; the stand-in object detector is off unless --standin, so the camera is the only eye.

uv run --python 3.12 --with mujoco --with opencv-python-headless==5.0.0.93 python eval/vision_walk.py [--seeds 8]
  [--stickers] [--standin] [--decline]
Default: no markers on the walls, the camera has only the room colours (zero stickers). --stickers: ArUco plates on 3 walls.
--decline: the person says no.
Out: out/vision/<house>/<setup>/<arm>/seed<N>/trace.jsonl + summary.json, and out/vision/<house>/<setup>/summary.json.
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import perceive  # noqa: E402
from agent.localize import ParticleFilter, _wrap  # noqa: E402
from agent.loop import follow, route  # noqa: E402
from agent.mission import ARRIVE, ARRIVE_SURE, GOAL_S, HZ, MARGIN, SQUEEZE, SURE, TURN, VX, goals  # noqa: E402
from agent.twin_sim import TwinSim  # noqa: E402
from agent.walk import LOST_S, colour_doubt, lost_tick, vision_decision  # noqa: E402


def episode(args):
    house_dir, seed, camera, stickers, standin, decline, out = args
    sim = TwinSim(house_dir, seed=seed, stickers=stickers)
    rng = np.random.default_rng(seed)
    hmap, rooms = sim.world, [r["name"] for r in sim.house["rooms"]]
    room_name = lambda x, y: rooms[k] if (k := int(hmap.room(x, y))) >= 0 else ""  # noqa: E731
    todo = goals(sim)
    k = int(rng.integers(len(todo))) if todo else 0
    todo = todo[k:] + todo[:k]
    true = start = sim.random_pose()  # before the colour survey, so both arms start in the same spot
    if camera:
        for r, h in zip(hmap.rooms, sim.room_colours()):
            if h:
                r["colour_hs"] = h
    pf = ParticleFilter.anywhere(hmap, seed=seed)
    t, trace, results, found, back, stopped, votes = 0.0, [], [], None, 0, False, []
    for g in todo:
        t0, path, last_plan, goal_bumps, lost, doubt = t, None, -9.0, 0, 0.0, -1.0
        while t - t0 < GOAL_S and goal_bumps < 12 and not stopped:
            pf.update_tof(*sim.tof(*true))
            if standin:
                pf.update_landmarks(sim.objects(*true), sim.hfov)
            before, seen, vote = (pf.estimate(), pf.confidence()), [], None
            if camera:
                seen, vote = perceive.see(pf, cv.cvtColor(sim.image(*true), cv.COLOR_RGB2BGR), sim.K)
            est, conf = pf.estimate(), pf.confidence()
            votes = (votes + [vote])[-5 * HZ:]
            if camera and (ev := vision_decision(hmap, room_name, g["xy"], before, (est, conf), seen, vote)):
                trace.append({"t": round(t, 1), **ev, "goal": g["name"], "true": [round(v, 3) for v in true]})
                last_plan = -9.0
            if found is None and conf >= SURE and math.hypot(est[0] - true[0], est[1] - true[1]) < 0.3:
                found = t
            mode, v, w, ahead = "look", 0.0, 0.6, None
            if back:
                mode, v, w, back = "back off", -0.1, 0.0, back - 1
            elif conf >= SURE:
                at = math.hypot(est[0] - g["xy"][0], est[1] - g["xy"][1]) < ARRIVE and conf >= ARRIVE_SURE and t > doubt
                q = at and camera and colour_doubt(votes, g["name"])
                if q:
                    trace.append({"t": round(t, 1), "event": "approval", "question": q, "answer": not decline, "by": "script",
                                  "cause": "room colour", "goal": g["name"]})
                    at, doubt = not decline, t if not decline else t + 10
                if at:
                    results.append({"goal": g["name"], "t": round(t - t0, 1), "declared": True,
                                    "true_dist": round(math.hypot(true[0] - g["xy"][0], true[1] - g["xy"][1]), 3)})
                    trace.append({"t": round(t, 1), "event": "arrive", **results[-1]})
                    break
                ahead, reached = follow(path, est) if path else (None, False)
                if reached or t - last_plan > 2.0:
                    path, last_plan = route(hmap, est[:2], g["xy"], MARGIN) or route(hmap, est[:2], g["xy"], SQUEEZE), t
                mode = "go" if path else "stuck"
                if path:
                    if last_plan == t:
                        ahead = follow(path, est)[0]
                    e = float(_wrap(math.atan2(ahead[1] - est[1], ahead[0] - est[0]) - est[2]))
                    w = max(-TURN, min(TURN, 2.0 * e))
                    v = VX * max(0.0, math.cos(e)) ** 3
                    if sim.clearance[int((est[1] - hmap.o[1]) / 0.02) + 1, int((est[0] - hmap.o[0]) / 0.02) + 1] < MARGIN + 0.02:
                        v = min(v, 0.08)
            lost += lost_tick(pf, mode)
            if lost > LOST_S:
                q = (f"Lost for {LOST_S} s ({conf:.0%} sure, best guess: {room_name(est[0], est[1]) or '?'}, the camera's colours "
                     f"say: {vote or 'no camera'}). Keep looking for the {g['name']}?")
                trace.append({"t": round(t, 1), "event": "approval", "question": q, "answer": not decline, "by": "script"})
                lost, stopped = 0.0, decline
                if stopped:
                    mode, v, w = "stop", 0.0, 0.0
            true, odo, bumped = sim.move(true, v, w, 1 / HZ)
            if bumped and v > 0:
                back, last_plan, goal_bumps = 3, -9.0, goal_bumps + 1
            pf.predict(*odo)
            trace.append({"t": round(t, 1), "true": [round(x, 3) for x in true], "est": [round(x, 3) for x in est], "conf": round(conf, 2),
                          "mode": mode, "goal": g["name"], "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1]),
                          "aruco": seen, "colour": vote})
            t += 1 / HZ
        else:
            results.append({"goal": g["name"], "t": round(t - t0, 1), "declared": False,
                            "why": "declined" if stopped else "stuck" if goal_bumps >= 12 else "time"})
        if stopped:
            break
    summary = {"seed": seed, "start": [round(x, 3) for x in start], "goals": len(todo),
               "reached": sum(r["declared"] and r["true_dist"] < 0.5 for r in results),
               "found_s": round(found, 1) if found is not None else None, "time_s": round(t, 1),
               "vision_decisions": sum(s.get("event") == "vision_decision" for s in trace),
               "approvals": [s["answer"] for s in trace if s.get("event") == "approval"], "results": results}
    d = Path(out) / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.jsonl").write_text("\n".join(json.dumps(s) for s in trace))
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(prog="vision_walk")
    ap.add_argument("--house", default=str(ROOT / "out/current"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--stickers", action="store_true")
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--decline", action="store_true")
    a = ap.parse_args()
    house_dir = Path(a.house).resolve()
    out = ROOT / "out/vision" / house_dir.name / ("_".join(k for k in ("stickers", "standin", "decline") if getattr(a, k)) or "default")
    arms = {"tof": False, "tof+camera": True}
    jobs = [(house_dir, s, cam, a.stickers, a.standin, a.decline, out / arm) for arm, cam in arms.items() for s in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        eps = list(pool.map(episode, jobs))
    total = {}
    for i, arm in enumerate(arms):
        e = eps[i * a.seeds:(i + 1) * a.seeds]
        fs = [x["found_s"] for x in e if x["found_s"] is not None]
        total[arm] = {"goals_reached": f'{sum(x["reached"] for x in e)}/{sum(x["goals"] for x in e)}',
                      "found": f"{len(fs)}/{len(e)}", "found_s_median": float(np.median(fs)) if fs else None,
                      "vision_decisions": sum(x["vision_decisions"] for x in e), "approvals": sum(len(x["approvals"]) for x in e),
                      "stopped": sum(any(r.get("why") == "declined" for r in x["results"]) for x in e)}
    total["setup"] = {"stickers": a.stickers, "standin": a.standin, "decline": a.decline, "seeds": a.seeds}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(total, indent=1))
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()
