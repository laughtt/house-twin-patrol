"""The 8x8 ToF: only its two level rows (3 and 4, 2.8° up and down) hit what the 20 cm map holds, so a frame becomes 8
zone ranges, scored against the map from every guess."""
import numpy as np

from agent.loc import TOF_AZ


def read(dist_mm, status, rowfix, rmax):
    """→ meas (2, 8) m, valid (2, 8), known (8: the sensor covers the zone; status 254 = no data), zones (8: what
    `score` compares; rmax + 1 = nothing in range) and ranges (8: measured zones only, nan elsewhere).
    Without rowfix a zone averages an invalid row as "far": one row at 1 m and one with no return made ~3 m. With it a
    zone is read from the rows that measured it (KISS-ICP drops no-return beams the same way)."""
    meas = np.array(dist_mm, float).reshape(8, 8)[3:5] / 1000.0
    st = np.array(status).reshape(8, 8)[3:5]
    valid = (st == 5) & np.isfinite(meas)
    n = valid.sum(0)
    ranges = np.where(n > 0, np.where(valid, meas, 0).sum(0) / np.maximum(n, 1), np.nan)
    zones = np.where(n > 0, ranges, rmax + 1.0) if rowfix else np.where(valid, meas, rmax + 1.0).mean(0)
    return meas, valid, (st != 254).any(0), zones, ranges


def score(house_map, p, zones, known, rmax):
    """How well each guess (x, y, yaw) explains the 8 zones: a Gaussian that widens with range, 0.1 for things not on
    the map, tempered (8 zones are not independent evidence)."""
    x, y, yaw = (np.repeat(p[:, k:k + 1], 8, 1) for k in range(3))
    exp = house_map.raycast(x, y, yaw + TOF_AZ, rmax=rmax)
    sigma = 0.05 + 0.03 * np.minimum(exp, rmax)
    both_far = (exp > rmax) & (zones > rmax)
    like = np.where(both_far | ~known, 1.0, np.exp(-0.5 * ((zones - exp) / sigma) ** 2)) * 0.9 + 0.1
    return np.exp(np.log(like).sum(1) / 4)
