"""Where is the duck? A particle filter on the house map (house.json frame: metres, z up). The jobs live in agent/loc/;
this runs them each tick:

odometry → predict (and the spin stack's turn) · ToF → tof.read → tof.score → weights; a full turn on the spot →
spin.match over the whole map → guesses re-seeded there · camera → landmarks (objects, room colour, ArUco) ·
estimate → fit.check → confidence, capped while the spin match or the fit check says "not sure yet".

Settings (all off = the filter before spin match, eval/selfloc.py's "base"): rowfix (zones from the rows that measured
them), spin (not sure until a full turn matches one place), fit (verify the estimate against the ToF), relook (match
again when the fit stays poor), tof_max (the sensor's range), jump (a sure duck doesn't teleport: an estimate far from
the last trusted pose, carried by odometry, is a false positive; the guesses still near that pose are the answer, and
if none are left the duck looks again, nearby first).
"""
import math

import numpy as np

from agent.loc import CELL, TOF_AZ, TOF_MAX, _wrap, landmarks, tof  # noqa: F401 - re-exported for callers
from agent.loc.fit import Fit
from agent.loc.housemap import HouseMap  # noqa: F401 - re-exported
from agent.loc.spin import Spin


class ParticleFilter:
    def __init__(self, house_map, x, y, yaw, n=400, spread=(0.05, 0.05), seed=0, rowfix=True, spin=True, fit=True,
                 relook=True, tof_max=TOF_MAX, jump=True):
        self.map, self.n, self.rng = house_map, n, np.random.default_rng(seed)
        self.p = np.c_[x + self.rng.normal(0, spread[0], n), y + self.rng.normal(0, spread[0], n), yaw + self.rng.normal(0, spread[1], n)]
        self.w = np.full(n, 1 / n)
        self.misfit = {}  # per landmark class: trusted sightings in a row that no guess explained
        self.w_slow = self.w_fast = None  # long- and short-term average ToF fit: a sudden drop = picked up and moved
        self.last_tof = None
        self.blind = 0  # ToF frames in a row with no target at all
        self.rowfix, self.relook, self.tof_max = rowfix, relook, tof_max
        self.spin = Spin(matched=True) if spin else None  # a known start needs no match
        self.fit = Fit() if fit or relook else None
        self.cap_fit, self.ranges, self.est = fit, None, None
        self.jump = jump and spin and fit
        self.anchor, self.walked = None, 0.0  # the last pose the fit check trusted, carried by odometry; metres since

    @classmethod
    def anywhere(cls, house_map, n=2000, seed=0, **settings):
        """Unknown start (switched on, or picked up and put down): guesses over every free cell of every room, any heading."""
        pf = cls(house_map, 0.0, 0.0, 0.0, n=n, seed=seed, **settings)
        pf.p = pf._random_poses(n)
        if pf.spin:
            pf.spin.matched = False
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
        self.p[:, 0] += f * c - l * s
        self.p[:, 1] += f * s + l * c
        # a duck can't stand inside furniture. (Holding guesses back at walls instead made the cloud lag and split on
        # routes that pass close to furniture: 65 % → 89 % of the time within 0.3 m when that rule went.)
        self.w *= self.map.free(self.p[:, 0], self.p[:, 1]) * 0.99 + 0.01
        self._normalise()
        if self.spin:
            self.spin.odometry(forward, left, turn)
        if self.anchor is not None:
            x, y, a = self.anchor
            self.anchor = (x + forward * math.cos(a) - left * math.sin(a), y + forward * math.sin(a) + left * math.cos(a), a + turn)
            self.walked += d

    def update_tof(self, dist_mm, status):
        """The two level rows against the map. Status 254 = no data for that zone (an emulated sensor that does not
        cover it), which is skipped rather than read as "nothing in range"."""
        meas, valid, known, zones, self.ranges = tof.read(dist_mm, status, self.rowfix, self.tof_max)
        frame = np.r_[meas.ravel(), valid.ravel()]
        frozen = self.last_tof is not None and np.array_equal(frame, self.last_tof)
        self.last_tof = frame
        self.blind = 0 if valid.any() else self.blind + 1
        # stuck, covered, or no target for 5 s (open space says "nothing within 4 m" for a moment; a dead sensor for ever)
        if frozen or self.blind > 25 or (valid.any() and (meas[valid] < 0.05).all()):
            self.ranges = np.full(8, np.nan)  # nothing the fit check can verify either
            return
        fit = tof.score(self.map, self.p, zones, known, self.tof_max)
        avg = float(np.sum(self.w * fit))
        self.w_slow = avg if self.w_slow is None else self.w_slow + 0.01 * (avg - self.w_slow)
        self.w_fast = avg if self.w_fast is None else self.w_fast + 0.1 * (avg - self.w_fast)
        self.w *= fit
        self._normalise()
        lost = 1 - self.w_fast / max(self.w_slow, 1e-12)
        # walls in view that stopped matching (not just open space): try fresh guesses; while a trusted pose is held,
        # relook matches again instead, nearby first
        if lost > 0.6 and valid.any(0).sum() >= 5 and self.anchor is None:
            k = self.rng.choice(self.n, max(1, int(self.n * min(0.02, 0.05 * lost))), replace=False)
            self.p[k] = self._random_poses(len(k))
            self.w[k] = self.w.mean()
        if self.spin and not self.spin.matched and self.spin.add(self.ranges, known & ~valid.any(0)):
            self.p = Spin.seeds(self.spin.match(self.map, self.tof_max), self.n, self.rng)
            self.w, self.last, self.anchor, self.spin.near = np.full(self.n, 1 / self.n), None, None, None
            if self.fit:  # the old fits judged the guesses before the match
                self.fit.hist = []

    def update_landmarks(self, dets, hfov):
        """Objects the camera recognised: [{cls, bearing (rad, + = left), range (m) or None}] (agent/loc/landmarks.py)."""
        landmarks.update(self, dets, hfov)

    def update_room(self, hist, sharp=25.0):
        landmarks.room(self, hist, sharp)

    def update_marker(self, marker_id, solutions):
        """One ArUco sighting (perceive.see passes those within 3 m). Once sure, its pose is the trusted one, so the jump
        check doesn't drop it. Not before: IPPE's two poses are told apart by the current guess, which is still random
        then (ending the spin on a marker: 48 → 32 of 48 goals, eval/vision_walk.py)."""
        pose = landmarks.marker(self, marker_id, solutions)
        if self.jump and self.spin.matched:
            self.anchor, self.walked = pose, 0.0
        return pose

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
        if self.jump and self.spin.matched and self.anchor is not None:
            home = np.hypot(self.p[:, 0] - self.anchor[0], self.p[:, 1] - self.anchor[1]) < self._gate()
            if math.hypot(c[0] - self.anchor[0], c[1] - self.anchor[1]) > self._gate() and self.w[home].sum() > 0.05:
                self.w *= np.where(home, 1.0, 0.01)  # a far cluster won: a false positive, its guesses mostly dropped
                self._normalise()
                c = np.average(self.p[:, :2], axis=0, weights=self.w * home + 1e-12)
        w = self.w * near(c) + 1e-12
        self.last = (float(np.average(self.p[:, 0], weights=w)), float(np.average(self.p[:, 1], weights=w)))
        self.est = (*self.last, math.atan2(np.sum(w * np.sin(self.p[:, 2])), np.sum(w * np.cos(self.p[:, 2]))))
        return self.est

    def confidence(self):
        """Share of the weight within 50 cm of the last estimate: near 1 = sure where it is, low = several places fit.
        Capped at 0.5 until the spin match is clear, and by the fit check (fit.cap)."""
        if getattr(self, "last", None) is None:
            self.estimate()
        c = float(self.w[np.hypot(self.p[:, 0] - self.last[0], self.p[:, 1] - self.last[1]) < 0.5].sum())
        if self.fit and self.ranges is not None and self.est is not None:
            self.fit.check(self.map, self.est, self.ranges, self.tof_max)
            self.ranges = None  # once per ToF frame
            if self.relook and self.spin and self.spin.matched and self.fit.lost():
                self._look_again()
            elif self.jump and self.spin.matched and self.anchor is not None and \
                    math.hypot(self.est[0] - self.anchor[0], self.est[1] - self.anchor[1]) > self._gate():
                self._look_again()  # no guess left near the trusted pose: carried off, or every guess went wrong
            elif self.jump and self.spin.matched and self.fit.cap() == 1.0:
                self.anchor, self.walked = self.est, 0.0
        if self.fit and self.cap_fit:
            c = min(c, self.fit.cap())
        return min(c, 0.5) if self.spin and not self.spin.matched else c

    def _gate(self):
        """How far the estimate may be from the trusted pose carried by odometry (8 % slip, plus a margin)."""
        return 1.0 + 0.1 * self.walked

    def _look_again(self):
        """Turn and match again: near the trusted pose first when one is held, the whole house otherwise."""
        self.spin.matched, self.spin.tries, self.fit.hist = False, 0, []
        self.spin.near = (*self.anchor[:2], 2.0 + 0.1 * self.walked) if self.anchor is not None else None
        self.spin.restart()

    def _normalise(self):
        total = self.w.sum()
        self.w = self.w / total if total > 0 and np.isfinite(total) else np.full(self.n, 1 / self.n)
        if 1 / np.sum(self.w ** 2) < self.n / 2:  # systematic resampling
            idx = np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(self.n)) / self.n).clip(0, self.n - 1)
            self.p, self.w = self.p[idx], np.full(self.n, 1 / self.n)
