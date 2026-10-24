"""Where is the duck? A particle filter on the house map (house.json frame: metres, z up).

Odometry moves the guesses; the 8x8 ToF (its two level rows) scores them against the map; recognised objects (fixed
ones count most) and the room's colour pick the right place even from an unknown start; an ArUco marker, if any, snaps them.
"""
import math

import cv2 as cv
import numpy as np

CELL = 0.02
TOF_AZ = -np.radians((np.arange(8) + 0.5) / 8 * 45 - 22.5)  # zone columns, col 0 = the sensor's left (+)
TOF_MAX = 4.0


class HouseMap:
    """What a sensor 20 cm above the floor can hit: walls, solid furniture, and only the legs of tables and chairs."""

    def __init__(self, house, eye_h=0.2):
        xs = [p for w in house["walls"] for p in (w["a"][0], w["b"][0])]
        ys = [p for w in house["walls"] for p in (w["a"][1], w["b"][1])]
        self.o = np.minimum(0.0, [min(xs), min(ys)])  # grid origin: (0, 0), or lower if a room lies west or south of it
        self.W, self.L = max(xs) - self.o[0], max(ys) - self.o[1]
        self.grid = np.zeros((int(self.L / CELL) + 3, int(self.W / CELL) + 3), bool)
        self.grid[:, :1] = self.grid[:, -2:] = self.grid[:1, :] = self.grid[-2:, :] = True
        for w in house["walls"]:  # every wall, interior ones too, as thick as it is
            a, b = (((np.array(w[k]) - self.o) / CELL + 1).astype(int) for k in ("a", "b"))
            self.grid |= cv.line(np.zeros(self.grid.shape, np.uint8), tuple(int(v) for v in a), tuple(int(v) for v in b), 1,
                                 max(1, int(round(w["t"] / CELL)))).astype(bool)
        for o in house["objects"]:
            (cx, cy), (w, d, h), yaw = o["center"], o["size"], o["yaw"]
            legs = o["kind"] == "table" or o["label"] == "chair"
            if h < eye_h and not legs:
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            parts = [(sx * (w / 2 - 0.04), sy * (d / 2 - 0.04), 0.02, 0.02) for sx in (-1, 1) for sy in (-1, 1)] if legs else [(0, 0, w / 2, d / 2)]
            for px, py, hw, hd in parts:
                u, v = np.meshgrid(np.arange(-hw, hw + CELL, CELL / 2), np.arange(-hd, hd + CELL, CELL / 2))
                x, y = cx + (px + u) * c - (py + v) * s - self.o[0], cy + (px + u) * s + (py + v) * c - self.o[1]
                ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.L)
                self.grid[(y[ok] / CELL).astype(int) + 1, (x[ok] / CELL).astype(int) + 1] = True
        self.markers = {m["id"]: m for m in house.get("markers", [])}
        self.landmarks = [lm for lm in house.get("landmarks", []) if lm.get("cls") and lm.get("weight", 0) > 0]
        self.rooms = house.get("rooms", [])
        self.room_of = np.full(self.grid.shape, -1, np.int16)  # which room each cell is in (-1: outside every room)
        for k, r in enumerate(self.rooms):
            for x0, y0, x1, y1 in r["rects"]:
                (x0, y0), (x1, y1) = np.subtract([x0, y0], self.o), np.subtract([x1, y1], self.o)
                self.room_of[int(y0 / CELL) + 1:int(y1 / CELL) + 2, int(x0 / CELL) + 1:int(x1 / CELL) + 2] = k

    def room(self, x, y):
        i = np.clip(((np.asarray(y) - self.o[1]) / CELL).astype(int) + 1, 0, self.grid.shape[0] - 1)
        j = np.clip(((np.asarray(x) - self.o[0]) / CELL).astype(int) + 1, 0, self.grid.shape[1] - 1)
        return self.room_of[i, j]

    def free(self, x, y):
        i, j = ((np.asarray(y) - self.o[1]) / CELL).astype(int) + 1, ((np.asarray(x) - self.o[0]) / CELL).astype(int) + 1
        inside = (i >= 0) & (i < self.grid.shape[0]) & (j >= 0) & (j < self.grid.shape[1])
        return inside & ~self.grid[np.clip(i, 0, self.grid.shape[0] - 1), np.clip(j, 0, self.grid.shape[1] - 1)]

    def raycast(self, x, y, angle, step=0.02):
        """Distance to the first hit for many rays at once (TOF_MAX + 1 when nothing within range)."""
        dist = np.full(np.shape(x), TOF_MAX + 1.0)
        alive = np.ones(np.shape(x), bool)
        for r in np.arange(0.05, TOF_MAX + step, step):
            hit = alive & ~self.free(x + r * np.cos(angle), y + r * np.sin(angle))
            dist[hit] = r
            alive &= ~hit
            if not alive.any():
                break
        return dist


class ParticleFilter:
    def __init__(self, house_map, x, y, yaw, n=400, spread=(0.05, 0.05), seed=0):
        self.map, self.n, self.rng = house_map, n, np.random.default_rng(seed)
        self.p = np.c_[x + self.rng.normal(0, spread[0], n), y + self.rng.normal(0, spread[0], n), yaw + self.rng.normal(0, spread[1], n)]
        self.w = np.full(n, 1 / n)
        self.misfit = {}  # per landmark class: trusted sightings in a row that no guess explained
        self.w_slow = self.w_fast = None  # long- and short-term average ToF fit: a sudden drop = picked up and moved
        self.last_tof = None

    @classmethod
    def anywhere(cls, house_map, n=2000, seed=0):
        """Unknown start (switched on, or picked up and put down): guesses over every free cell of every room, any heading."""
        pf = cls(house_map, 0.0, 0.0, 0.0, n=n, seed=seed)
        pf.p = pf._random_poses(n)
        return pf

    def _random_poses(self, k):
        cells = np.argwhere(~self.map.grid & (self.map.room_of >= 0))
        if not len(cells):  # rooms that cover no free cell: anywhere free
            cells = np.argwhere(~self.map.grid)
        pick = cells[self.rng.integers(len(cells), size=k)]
        return np.c_[(pick[:, 1] - 1 + self.rng.random(k)) * CELL + self.map.o[0], (pick[:, 0] - 1 + self.rng.random(k)) * CELL + self.map.o[1],
                     self.rng.uniform(-np.pi, np.pi, k)]

    def predict(self, forward, left, turn):
        """Odometry deltas in the duck's frame since the last call; noise covers its slip and heading drift."""
        if not np.isfinite([forward, left, turn]).all():
            return
        d = math.hypot(forward, left)
        f = forward + self.rng.normal(0, 0.08 * d + 0.002, self.n)
        l = left + self.rng.normal(0, 0.04 * d + 0.002, self.n)
        self.p[:, 2] += turn + self.rng.normal(0, 0.06 * abs(turn) + 0.08 * d + 0.002, self.n)
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        x, y = self.p[:, 0] + f * c - l * s, self.p[:, 1] + f * s + l * c
        ok = self.map.free(x, y)  # a duck can't go through walls or furniture: it stays put (wheels slip)
        self.p[ok, 0], self.p[ok, 1] = x[ok], y[ok]
        self.w *= np.where(ok, 1.0, 0.3)
        self._normalise()

    def update_tof(self, dist_mm, status):
        """The two level rows (3 and 4) against the map: 16 ranges per particle. Status 254 = no data for that zone
        (an emulated sensor that does not cover it), which is skipped rather than read as "nothing in range"."""
        meas = np.array(dist_mm, float).reshape(8, 8)[3:5] / 1000.0
        valid = (np.array(status).reshape(8, 8)[3:5] == 5) & np.isfinite(meas)
        known = (np.array(status).reshape(8, 8)[3:5] != 254).any(0)
        frame = np.r_[meas.ravel(), valid.ravel()]
        frozen = self.last_tof is not None and np.array_equal(frame, self.last_tof)
        self.last_tof = frame
        if frozen or not valid.any() or (meas[valid] < 0.05).all():  # stuck, blind ("nothing within 4 m" indoors) or covered
            return
        meas = np.where(valid, meas, TOF_MAX + 1.0).mean(0)  # rows 3 and 4 look 2.8° up and down: average them
        x, y, yaw = (np.repeat(self.p[:, k:k + 1], 8, 1) for k in range(3))
        exp = self.map.raycast(x, y, yaw + TOF_AZ)
        sigma = 0.05 + 0.03 * np.minimum(exp, TOF_MAX)
        both_far = (exp > TOF_MAX) & (meas > TOF_MAX)
        like = np.where(both_far | ~known, 1.0, np.exp(-0.5 * ((meas - exp) / sigma) ** 2)) * 0.9 + 0.1  # 0.1: things not on the map
        fit = np.exp(np.log(like).sum(1) / 4)  # tempered: 8 zones are not independent evidence
        avg = float(np.sum(self.w * fit))
        self.w_slow = avg if self.w_slow is None else self.w_slow + 0.01 * (avg - self.w_slow)
        self.w_fast = avg if self.w_fast is None else self.w_fast + 0.1 * (avg - self.w_fast)
        self.w *= fit
        self._normalise()
        lost = 1 - self.w_fast / max(self.w_slow, 1e-12)
        if lost > 0.6:  # the walls stopped matching where we think we are: try fresh guesses anywhere (augmented MCL)
            k = self.rng.choice(self.n, max(1, int(self.n * min(0.02, 0.05 * lost))), replace=False)
            self.p[k] = self._random_poses(len(k))
            self.w[k] = self.w.mean()

    def update_landmarks(self, dets, hfov):
        """Objects the camera recognised: [{cls, bearing (rad, + = left), range (m) or None}]. Each scores every guess by
        the mapped landmark of that class that best explains it (in view, not behind a wall), times how much that kind
        of thing stays put (house.json weight: fixed 1, rarely moved 0.6, often moved 0.2)."""
        x, y, yaw = self.p.T
        dets = [{**d, "range": d.get("range") if d.get("range") is not None and np.isfinite(d["range"]) and d["range"] > 0 else None}
                for d in dets if np.isfinite(d.get("bearing", np.nan))]
        for d in sorted(dets, key=lambda d: -d.get("score", 0))[:8]:  # a cluttered frame: the 8 surest sightings
            best = np.zeros(self.n)
            for lm in self.map.landmarks:
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
                    seen = self.map.raycast(x[k], y[k], yaw[k] + b[k]) >= np.minimum(r[k] - rad - 0.3, TOF_MAX)
                    best[k] = np.maximum(best[k], lm["weight"] * like[k] * seen)
            fixed = [lm for lm in self.map.landmarks if lm["cls"] == d["cls"] and lm["weight"] >= 0.6]
            if fixed:  # trusted sightings that no guess explains, three in a row: the guesses are probably all wrong
                c = d["cls"]
                self.misfit[c] = self.misfit.get(c, 0) + 1 if np.sum(self.w * best) < 0.1 else 0
                if self.misfit[c] >= 3:
                    k = self._reseed(d, fixed)
                    best[k] = 1.0
                    x, y, yaw = self.p.T
                    self.misfit[c] = 0
            self.w *= 0.2 + 0.8 * best  # 0.2: the detector can be wrong, or the thing was moved
        self._normalise()

    def _reseed(self, d, fixed, share=0.05):
        """Replace a share of the guesses with poses that would see one of these landmarks this way (sensor resetting)."""
        k = self.rng.choice(self.n, max(1, int(self.n * share)), replace=False)
        lm = [fixed[i] for i in self.rng.integers(len(fixed), size=len(k))]
        c = np.array([m["center"] for m in lm])
        r = (np.full(len(k), d["range"]) + 0.7 * np.array([max(m["size"][:2]) / 2 for m in lm]) if d.get("range") is not None
             else self.rng.uniform(0.5, 4.0, len(k))) * self.rng.normal(1, 0.1, len(k))
        yaw = self.rng.uniform(-np.pi, np.pi, len(k))
        a = yaw + d["bearing"] + self.rng.normal(0, np.radians(4), len(k))
        self.p[k] = np.c_[c[:, 0] - r * np.cos(a), c[:, 1] - r * np.sin(a), yaw]
        self.w[k] = self.w.mean()
        return k

    def update_room(self, hist, sharp=25.0):
        """The camera's colour fingerprint against each room's (house.json colour_hs): a weak vote for the room."""
        sims = np.array([np.sum(np.sqrt(np.asarray(hist) * r["colour_hs"])) if "colour_hs" in r else np.nan for r in self.map.rooms])
        if not np.isfinite(sims).any():
            return
        score = np.r_[np.nan_to_num(np.exp(sharp * (sims - np.nanmax(sims))), nan=1.0), 0.5]  # last: outside every room
        self.w *= 0.6 + 0.4 * score[self.map.room(self.p[:, 0], self.p[:, 1])]
        self._normalise()

    def update_marker(self, marker_id, solutions):
        """One ArUco sighting. IPPE gives two camera poses; keep the one nearest the current guess."""
        m = self.map.markers[marker_id]
        n = np.array([math.cos(m["yaw"]), math.sin(m["yaw"]), 0.0])
        M = np.c_[[-n[1], n[0], 0.0], [0.0, 0.0, 1.0], n]  # marker axes in the house: right, up, out of the face
        est = self.estimate()
        poses = []
        for R, t, _ in solutions:
            cam = np.array([m["x"], m["y"], m["z"]]) + M @ (-R.T @ t)
            fwd = M @ (R.T @ [0.0, 0.0, 1.0])
            poses.append((cam[0], cam[1], math.atan2(fwd[1], fwd[0]), float(np.linalg.norm(t))))
        x, y, yaw, rng = min(poses, key=lambda q: math.hypot(q[0] - est[0], q[1] - est[1]) + 0.3 * abs(_wrap(q[2] - est[2])))
        sxy, syaw = 0.03 + 0.04 * rng, math.radians(3 + 2 * rng)
        d2 = ((self.p[:, 0] - x) ** 2 + (self.p[:, 1] - y) ** 2) / sxy ** 2 + _wrap(self.p[:, 2] - yaw) ** 2 / syaw ** 2
        self.w *= np.exp(-0.5 * d2) + 1e-6
        self._normalise()
        if 1 / np.sum(self.w ** 2) < self.n / 10:  # a sighting far from every guess: re-seed a share of them around it
            k = self.n // 3
            self.p[:k] = np.c_[self.rng.normal(x, sxy, k), self.rng.normal(y, sxy, k), self.rng.normal(yaw, syaw, k)]
            self.w[:k] = self.w.max()
            self._normalise()
        return x, y, yaw

    def estimate(self):
        """The strongest group of guesses (30 cm cells, then everything within 50 cm of the best one): stray guesses
        elsewhere in the house don't drag the answer between places."""
        cell = np.floor(self.p[:, :2] / 0.3).astype(np.int64)
        uk, inv = np.unique(cell[:, 0] * 100003 + cell[:, 1], return_inverse=True)
        cw, nb = np.bincount(inv, self.w), np.zeros(len(uk))
        for off in (-100004, -100003, -100002, -1, 0, 1, 100002, 100003, 100004):  # each cell with its 8 neighbours
            j = np.searchsorted(uk, uk + off).clip(0, len(uk) - 1)
            nb += np.where(uk[j] == uk + off, cw[j], 0)
        top = inv == nb.argmax()
        c = np.average(self.p[top, :2], axis=0, weights=self.w[top] + 1e-12)
        near = lambda q: np.hypot(self.p[:, 0] - q[0], self.p[:, 1] - q[1]) < 0.5  # noqa: E731
        if getattr(self, "last", None) is not None and self.w[near(self.last)].sum() > 0.6 * self.w[near(c)].sum():
            c = self.last  # stay with the place we were at unless another one is clearly stronger
        w = self.w * near(c) + 1e-12
        self.last = (float(np.average(self.p[:, 0], weights=w)), float(np.average(self.p[:, 1], weights=w)))
        return (*self.last, math.atan2(np.sum(w * np.sin(self.p[:, 2])), np.sum(w * np.cos(self.p[:, 2]))))

    def _normalise(self):
        total = self.w.sum()
        self.w = self.w / total if total > 0 and np.isfinite(total) else np.full(self.n, 1 / self.n)
        if 1 / np.sum(self.w ** 2) < self.n / 2:  # systematic resampling
            idx = np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(self.n)) / self.n).clip(0, self.n - 1)
            self.p, self.w = self.p[idx], np.full(self.n, 1 / self.n)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
