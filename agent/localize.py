"""Where is the duck? A particle filter on the house map (house.json frame: metres, z up).

Odometry moves the guesses; the 8x8 ToF (its two level rows) scores them against the map; an ArUco marker snaps them.
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
        self.W, self.L = max(xs), max(ys)
        self.grid = np.zeros((int(self.L / CELL) + 3, int(self.W / CELL) + 3), bool)
        self.grid[:, :1] = self.grid[:, -2:] = self.grid[:1, :] = self.grid[-2:, :] = True
        for w in house["walls"]:  # every wall, interior ones too, as thick as it is
            a, b = ((np.array(w[k]) / CELL + 1).astype(int) for k in ("a", "b"))
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
                x, y = cx + (px + u) * c - (py + v) * s, cy + (px + u) * s + (py + v) * c
                ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.L)
                self.grid[(y[ok] / CELL).astype(int) + 1, (x[ok] / CELL).astype(int) + 1] = True
        self.markers = {m["id"]: m for m in house.get("markers", [])}

    def free(self, x, y):
        i, j = (np.asarray(y) / CELL).astype(int) + 1, (np.asarray(x) / CELL).astype(int) + 1
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

    def predict(self, forward, left, turn):
        """Odometry deltas in the duck's frame since the last call; noise covers its slip and heading drift."""
        d = math.hypot(forward, left)
        f = forward + self.rng.normal(0, 0.08 * d + 0.002, self.n)
        l = left + self.rng.normal(0, 0.04 * d + 0.002, self.n)
        self.p[:, 2] += turn + self.rng.normal(0, 0.06 * abs(turn) + 0.08 * d + 0.002, self.n)
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += f * c - l * s
        self.p[:, 1] += f * s + l * c
        self.w *= self.map.free(self.p[:, 0], self.p[:, 1]) * 0.99 + 0.01  # a duck can't stand inside furniture
        self._normalise()

    def update_tof(self, dist_mm, status):
        """The two level rows (3 and 4) against the map: 16 ranges per particle."""
        meas = np.array(dist_mm, float).reshape(8, 8)[3:5] / 1000.0
        valid = np.array(status).reshape(8, 8)[3:5] == 5
        meas = np.where(valid, meas, TOF_MAX + 1.0).mean(0)  # rows 3 and 4 look 2.8° up and down: average them
        x, y, yaw = (np.repeat(self.p[:, k:k + 1], 8, 1) for k in range(3))
        exp = self.map.raycast(x, y, yaw + TOF_AZ)
        sigma = 0.05 + 0.03 * np.minimum(exp, TOF_MAX)
        both_far = (exp > TOF_MAX) & (meas > TOF_MAX)
        like = np.where(both_far, 1.0, np.exp(-0.5 * ((meas - exp) / sigma) ** 2)) * 0.9 + 0.1  # 0.1: things not on the map
        self.w *= np.exp(np.log(like).sum(1) / 4)  # tempered: 8 zones are not independent evidence
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
        return (float(np.sum(self.w * self.p[:, 0])), float(np.sum(self.w * self.p[:, 1])),
                math.atan2(np.sum(self.w * np.sin(self.p[:, 2])), np.sum(self.w * np.cos(self.p[:, 2]))))

    def _normalise(self):
        total = self.w.sum()
        self.w = self.w / total if total > 0 and np.isfinite(total) else np.full(self.n, 1 / self.n)
        if 1 / np.sum(self.w ** 2) < self.n / 2:  # systematic resampling
            idx = np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(self.n)) / self.n).clip(0, self.n - 1)
            self.p, self.w = self.p[idx], np.full(self.n, 1 / self.n)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
