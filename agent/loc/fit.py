"""Fit check, verify before trust (ORB-SLAM3 checks a place-recognition hit geometrically before using it): from the
estimate, ray-cast the zones the sensor measured: the share that fit (15 cm at 1 m, 25 cm at 3 m) among those that can
tell. A dropout or a reading shorter than the map says is no evidence either way.
Over 5 s it says whether the guess is believable at all; over the last second, whether the duck may say "I'm here"
(in open space past the sensor's range nothing can be verified, and odometry drifts 0.5–1 m across a big room)."""
import numpy as np

from agent.loc import TOF_AZ

KEEP = 25  # frames (5 s at 5 Hz)
SURE_FIT, ARRIVE_FIT, LOST_FIT = 0.6, 0.7, 0.5


class Fit:
    def __init__(self):
        self.hist = []

    def check(self, house_map, est, ranges, rmax):
        ok = np.isfinite(ranges)
        f = None  # fewer than 3 zones that can tell (open space past the sensor's range, a dead sensor): can't verify
        if ok.any():
            exp = house_map.raycast(np.full(ok.sum(), est[0]), np.full(ok.sum(), est[1]), est[2] + TOF_AZ[ok], rmax=rmax)
            tol = 0.10 + 0.05 * np.minimum(exp, rmax)  # 15 cm at 1 m, 25 at 3: a walking duck's head wobbles a few degrees
            good, long_ = np.abs(ranges[ok] - exp) < tol, ranges[ok] > exp + tol
            # shorter than the map says is something not on it (a person, a bag; the floor when the walking duck's head
            # pitches down: Pollen's policy, live, hits it 1.5-2 m ahead): no evidence. Longer can't happen at the right pose.
            if good.sum() + long_.sum() >= 3:
                f = float(good.sum() / (good.sum() + long_.sum()))
        self.hist = (self.hist + [f])[-KEEP:]

    def cap(self):
        """The most confidence the fit allows: 0.5 (not sure: look around) / 0.8 (may drive, not declare) / 1."""
        seen, now = [h for h in self.hist if h is not None], [h for h in self.hist[-5:] if h is not None]
        if len(seen) < 5 or np.mean(seen) < SURE_FIT:
            return 0.5
        return 1.0 if now and np.mean(now) >= ARRIVE_FIT else 0.8  # "I'm here" only on a fresh fit: odometry drifts

    def lost(self):
        """Poor fit for 3 s: the lock went bad (kidnapped, slipped): match again."""
        seen = [h for h in self.hist if h is not None]
        return len(seen) >= 15 and np.mean(seen[-15:]) < LOST_FIT
