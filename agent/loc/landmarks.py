"""What the camera adds: recognised objects (fixed ones count most), the room's colour, and ArUco markers. Each scores
the guesses of a ParticleFilter `pf` in place."""
import math

import numpy as np

from agent.loc import TOF_MAX, _wrap


def update(pf, dets, hfov):
    """Objects the camera recognised: [{cls, bearing (rad, + = left), range (m) or None}]. Each scores every guess by
    the mapped landmark of that class that best explains it (in view, not behind a wall), times how much that kind
    of thing stays put (house.json weight: fixed 1, rarely moved 0.6, often moved 0.2)."""
    x, y, yaw = pf.p.T
    dets = [{**d, "range": d.get("range") if d.get("range") is not None and np.isfinite(d["range"]) and d["range"] > 0 else None}
            for d in dets if np.isfinite(d.get("bearing", np.nan))]
    for d in sorted(dets, key=lambda d: -d.get("score", 0))[:8]:  # a cluttered frame: the 8 surest sightings
        best = np.zeros(pf.n)
        for lm in pf.map.landmarks:
            if lm["cls"] != d["cls"]:
                continue
            dx, dy = lm["center"][0] - x, lm["center"][1] - y
            r, rad = np.hypot(dx, dy), max(lm["size"][:2]) / 2
            b = _wrap(np.arctan2(dy, dx) - yaw)
            like = np.exp(-0.5 * (_wrap(d["bearing"] - b) / (np.radians(4) + np.arctan2(rad, np.maximum(r, 0.3)))) ** 2)
            if d.get("range") is not None:
                like *= np.exp(-0.5 * ((d["range"] - np.maximum(r - 0.7 * rad, 0.1)) / (0.1 + 0.2 * r)) ** 2)
            like *= (np.abs(b) < hfov / 2 + np.arctan2(rad, np.maximum(r, 0.3))) & (r < 5.0)
            k = np.nonzero(like * lm["weight"] > best + 1e-3)[0]  # only guesses this landmark could improve
            if len(k):
                seen = pf.map.raycast(x[k], y[k], yaw[k] + b[k]) >= np.minimum(r[k] - rad - 0.3, TOF_MAX)
                best[k] = np.maximum(best[k], lm["weight"] * like[k] * seen)
        fixed = [lm for lm in pf.map.landmarks if lm["cls"] == d["cls"] and lm["weight"] >= 0.6]
        if fixed:  # trusted sightings that no guess explains, three in a row: the guesses are probably all wrong
            c = d["cls"]
            pf.misfit[c] = pf.misfit.get(c, 0) + 1 if np.sum(pf.w * best) < 0.1 else 0
            if pf.misfit[c] >= 3:
                k = reseed(pf, d, fixed)
                best[k] = 1.0
                x, y, yaw = pf.p.T
                pf.misfit[c] = 0
        pf.w *= 0.2 + 0.8 * best  # 0.2: the detector can be wrong, or the thing was moved
    pf._normalise()


def reseed(pf, d, fixed, share=0.05):
    """Replace a share of the guesses with poses that would see one of these landmarks this way (sensor resetting)."""
    k = pf.rng.choice(pf.n, max(1, int(pf.n * share)), replace=False)
    lm = [fixed[i] for i in pf.rng.integers(len(fixed), size=len(k))]
    c = np.array([m["center"] for m in lm])
    r = (np.full(len(k), d["range"]) + 0.7 * np.array([max(m["size"][:2]) / 2 for m in lm]) if d.get("range") is not None
         else pf.rng.uniform(0.5, 4.0, len(k))) * pf.rng.normal(1, 0.1, len(k))
    yaw = pf.rng.uniform(-np.pi, np.pi, len(k))
    a = yaw + d["bearing"] + pf.rng.normal(0, np.radians(4), len(k))
    pf.p[k] = np.c_[c[:, 0] - r * np.cos(a), c[:, 1] - r * np.sin(a), yaw]
    pf.w[k] = pf.w.mean()
    return k


def room(pf, hist, sharp=25.0):
    """The camera's colour fingerprint against each room's (house.json colour_hs): a weak vote for the room."""
    sims = np.array([np.sum(np.sqrt(np.asarray(hist) * r["colour_hs"])) if "colour_hs" in r else np.nan for r in pf.map.rooms])
    if not np.isfinite(sims).any():
        return
    score = np.r_[np.nan_to_num(np.exp(sharp * (sims - np.nanmax(sims))), nan=1.0), 0.5]  # last: outside every room
    pf.w *= 0.6 + 0.4 * score[pf.map.room(pf.p[:, 0], pf.p[:, 1])]
    pf._normalise()


def marker(pf, marker_id, solutions):
    """One ArUco sighting. IPPE gives two camera poses; keep the one nearest the current guess."""
    m = pf.map.markers[marker_id]
    n = np.array([math.cos(m["yaw"]), math.sin(m["yaw"]), 0.0])
    M = np.c_[[-n[1], n[0], 0.0], [0.0, 0.0, 1.0], n]  # marker axes in the house: right, up, out of the face
    est = pf.estimate()
    poses = []
    for R, t, _ in solutions:
        cam = np.array([m["x"], m["y"], m["z"]]) + M @ (-R.T @ t)
        fwd = M @ (R.T @ [0.0, 0.0, 1.0])
        poses.append((cam[0], cam[1], math.atan2(fwd[1], fwd[0]), float(np.linalg.norm(t))))
    x, y, yaw, rng = min(poses, key=lambda q: math.hypot(q[0] - est[0], q[1] - est[1]) + 0.3 * abs(_wrap(q[2] - est[2])))
    sxy, syaw = 0.03 + 0.04 * rng, math.radians(3 + 2 * rng)
    d2 = ((pf.p[:, 0] - x) ** 2 + (pf.p[:, 1] - y) ** 2) / sxy ** 2 + _wrap(pf.p[:, 2] - yaw) ** 2 / syaw ** 2
    pf.w *= np.exp(-0.5 * d2) + 1e-6
    pf._normalise()
    if 1 / np.sum(pf.w ** 2) < pf.n / 10:  # a sighting far from every guess: re-seed a share of them around it
        k = pf.n // 3
        pf.p[:k] = np.c_[pf.rng.normal(x, sxy, k), pf.rng.normal(y, sxy, k), pf.rng.normal(yaw, syaw, k)]
        pf.w[:k] = pf.w.max()
        pf._normalise()
    return x, y, yaw
