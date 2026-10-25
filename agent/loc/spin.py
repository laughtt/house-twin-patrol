"""Spin match: while the duck turns on the spot, its zone ranges are stacked into one 360° scan (3° bins). After a full
turn the scan is matched against the range profile of every free spot of the map at every heading (Cartographer's
rotational scan matcher), first on a 20 cm lattice, then at 10 cm around the best places. The match is clear when the
best place beats the next one (≥ 50 cm away) by MARGIN; until then the duck isn't sure and keeps turning."""
import math

import numpy as np

from agent.loc import TOF_AZ

B = 120         # 3° bins
MARGIN = 10.0   # log-likelihood: right matches were ≥ 15 ahead of the next place, wrong ones ≤ 5.6 (3 houses x 8 seeds)
MOVE = 0.10     # m from where the stack began: a scan stacked while driving smears; start over
PLACES = 8      # candidate places refined at 10 cm
TRIES = 3       # unclear turns before the filter is left to decide with the fit check (a room that looks like another)


class Spin:
    def __init__(self, matched=False):
        self.matched, self.tries = matched, 0
        self.near = None  # (x, y, radius): where the duck should be (last trusted pose + odometry); searched first
        self.restart()

    def restart(self):
        self.acc, self.far, self.turn, self.at = [[] for _ in range(B)], np.zeros(B, int), 0.0, np.zeros(2)

    def odometry(self, forward, left, turn):
        """Net displacement since the stack began, not path length: a walking policy turning on the spot sways its
        trunk ~60 cm a turn (Pollen's duck, live) and ends 2 cm from where it started."""
        c, s = math.cos(self.turn), math.sin(self.turn)
        self.at += (forward * c - left * s, forward * s + left * c)
        self.turn += turn
        if math.hypot(*self.at) > MOVE:
            self.restart()

    def add(self, ranges, far):
        """The 8 zone ranges (nan: not measured) and which zones said "no target" at the current heading; True once a
        full turn is stacked."""
        for a, r, f in zip(self.turn + TOF_AZ, ranges, far):
            b = int(round(a / (2 * np.pi / B))) % B
            if np.isfinite(r):
                self.acc[b].append(r)
            self.far[b] += f
        return abs(self.turn) >= 2 * np.pi

    def match(self, house_map, rmax):
        """→ places (k, 4): x, y, heading now, score; best first, ≥ 50 cm apart. Marks the match clear or counts a try."""
        xy, ij, prof = house_map.profiles(rmax)
        # a direction with no return twice and none measured is open space, evidence too (AMCL drops max-range beams
        # and keeps a TODO about it): with a 2.5 m sensor most bins of a room are empty and the match was left unsure
        scan = np.array([np.median(a) if a else rmax + 1.0 if f >= 2 else np.nan for a, f in zip(self.acc, self.far)])
        coarse = np.flatnonzero((ij[:, 0] % 2 == 0) & (ij[:, 1] % 2 == 0))
        best = score(prof[coarse], scan, rmax).max(1)
        order = coarse[np.argsort(-best)]
        seeds = pick(xy, order)
        if self.near:  # the best places nearby too, though the house may have better-looking ones elsewhere
            nx, ny, r = self.near
            seeds += [q for q in pick(xy, order[np.hypot(xy[order, 0] - nx, xy[order, 1] - ny) < r]) if q not in seeds]
        places = []
        for q in seeds:  # 10 cm around each: the 20 cm lattice can sit between a sharp peak's cells
            near = np.flatnonzero(np.abs(ij - ij[q]).max(1) <= 2)
            sc = score(prof[near], scan, rmax)
            k = np.unravel_index(sc.argmax(), sc.shape)
            places.append((*xy[near[k[0]]], k[1] * 2 * np.pi / B + self.turn, float(sc[k])))
        places = np.array(sorted(places, key=lambda m: -m[3]))
        keep = [places[0]] + [m for m in places[1:] if math.hypot(m[0] - places[0][0], m[1] - places[0][1]) >= 0.5]
        places = np.array(keep)
        if self.near:  # stay nearby unless the best place there is clearly worse than one elsewhere (carried off)
            local = places[np.hypot(places[:, 0] - nx, places[:, 1] - ny) < r]
            if len(local) and local[0, 3] >= places[0, 3] - MARGIN:
                places = local
        self.tries += 1
        self.matched = len(places) < 2 or places[0, 3] - places[1, 3] >= MARGIN or self.tries >= TRIES
        self.restart()
        return places

    @staticmethod
    def seeds(places, n, rng):
        """n guesses over every place within 2 MARGIN of the best, weighted by score, 5 cm / 3° of noise."""
        keep = places[places[:, 3] >= places[0, 3] - 2 * MARGIN]
        w = np.exp((keep[:, 3] - keep[0, 3]) / MARGIN)
        return keep[rng.choice(len(keep), n, p=w / w.sum()), :3] + rng.normal(0, [0.05, 0.05, 0.05], (n, 3))


def pick(xy, order):
    """The first PLACES spots of this order at least 50 cm apart."""
    seeds = []
    for p in order:
        if all(math.hypot(*(xy[p] - xy[q])) >= 0.5 for q in seeds):
            seeds.append(p)
        if len(seeds) == PLACES:
            break
    return seeds


def score(prof, scan, rmax):
    """Log-likelihood of the scan for each profile (spot) at each of the B headings: map bin = scan bin + k."""
    bins = np.flatnonzero(np.isfinite(scan))
    m = scan[bins].astype(np.float32)
    out = np.empty((len(prof), B), np.float32)
    for k in range(B):
        e = prof[:, (bins + k) % B]
        like = np.where((e > rmax) & (m > rmax), 1.0, np.exp(-0.5 * ((m - e) / (0.05 + 0.03 * np.minimum(e, rmax))) ** 2)) * 0.9 + 0.1
        out[:, k] = np.log(like).sum(1)
    return out
