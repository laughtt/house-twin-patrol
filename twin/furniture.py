"""What stands in the duck's way inside one space, from the LiDAR points themselves."""
import cv2 as cv
import numpy as np

from twin.plan.rooms import rects_of


def furniture(V, space, landmarks, cell=0.05):
    """A median top-height map below 1.2 m (upper cabinets don't count), cut where the height jumps (seats ~0.45,
    tables ~0.75, counters ~0.95), each layer as its largest rectangles; the nearest landmark names each one."""
    (x0, y0), (x1, y1) = np.array(space["polygon"]).min(0), np.array(space["polygon"]).max(0)
    inner = np.zeros_like(space["mask"])  # the outline (pushed to the walls), 5 cm clear of them: wall counters stay
    cv.fillPoly(inner, [np.round((np.array(space["polygon"]) - space["lo"]) / space["cell"]).astype(np.int32)], 1)
    inner = cv.erode(inner, np.ones((3, 3), np.uint8))
    ij = ((V[:, :2] - space["lo"]) / space["cell"]).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < inner.shape[1]) & (ij[:, 1] < inner.shape[0])
    sel = np.zeros(len(V), bool)
    sel[ok] = inner[ij[ok, 1], ij[ok, 0]] > 0
    low = V[sel & (V[:, 2] > 0.2) & (V[:, 2] < 1.2)]  # under 20 cm: rugs, thresholds, floor bumps, not furniture
    tall = V[sel & (V[:, 2] > 0.2) & (V[:, 2] < 2.0)]
    rows, cols = int((y1 - y0) / cell) + 1, int((x1 - x0) / cell) + 1
    key = ((low[:, 1] - y0) / cell).astype(int) * cols + ((low[:, 0] - x0) / cell).astype(int)
    order = np.lexsort((low[:, 2], key))
    cells, first, count = np.unique(key[order], return_index=True, return_counts=True)
    hmap = np.zeros((rows, cols), np.float32)
    hmap.flat[cells[count >= 2]] = low[order, 2][(first + count // 2)[count >= 2]]
    out = []
    for lo, hi, guess in ((0.2, 0.62, "seat"), (0.62, 0.86, "table"), (0.86, 1.21, "counter")):
        mask = ((hmap >= lo) & (hmap < hi)).astype(np.uint8)
        mask = cv.morphologyEx(cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        for r0, c0, r1, c1 in rects_of(mask.astype(bool), cell, 0.15, 0.3):
            bx0, bx1, by0, by1 = x0 + c0 * cell, x0 + c1 * cell, y0 + r0 * cell, y0 + r1 * cell
            bx0, by0 = (x0 if bx0 < x0 + 0.25 else bx0), (y0 if by0 < y0 + 0.25 else by0)  # flush to the wall it touches
            bx1, by1 = (x1 if bx1 > x1 - 0.25 else bx1), (y1 if by1 > y1 - 0.25 else by1)
            top = float(np.percentile(hmap[r0:r1, c0:c1][hmap[r0:r1, c0:c1] > 0], 90))
            near = [lm for lm in landmarks if bx0 - 0.3 < lm["center"][0] < bx1 + 0.3 and by0 - 0.3 < lm["center"][1] < by1 + 0.3]
            mid = ((bx0 + bx1) / 2, (by0 + by1) / 2)
            label = min(near, key=lambda lm: np.hypot(lm["center"][0] - mid[0], lm["center"][1] - mid[1]))["label"] if near else guess
            if label == "fridge":  # taller than the 1.2 m map
                foot = tall[(tall[:, 0] > bx0) & (tall[:, 0] < bx1) & (tall[:, 1] > by0) & (tall[:, 1] < by1)]
                top = float(np.percentile(foot[:, 2], 95)) if len(foot) else top
            out.append({"label": label, "kind": "table" if "table" in label else "box", "center": [(bx0 + bx1) / 2, (by0 + by1) / 2],
                        "size": [round(bx1 - bx0, 2), round(by1 - by0, 2), round(top, 2)], "yaw": 0.0})
    return out
