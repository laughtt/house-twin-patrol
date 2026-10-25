"""Stage 2, walls: vertical surfaces spanning at least 1 m of height in a cell. Counter fronts (0-0.9 m), sofa backs and
upper cabinets (1.4-2.2 m) are not walls; a door's lintel isn't either, so doorways stay open.
Out: the face map, `low` (any vertical surface below 60 cm: a sill, a radiator, a sofa side), straight wall lines, `solid` (bits of wall < 40 cm apart joined: one wall; it keeps rooms apart) and
`sealed` (the walls plus glass: collinear wall pieces bridged across gaps up to 3 m). Also `walls_from`: room outlines + doors → the map's walls."""
import cv2 as cv
import numpy as np

WALL_T = 0.10
DOOR_REACH = 0.8  # a door clears the walls this close to its line, across its width: the outlines of the two rooms it
                  # joins can stand apart (two walls with a strip between), and the centre can sit off the line
MIN_RUN = 8        # cells (40 cm): shorter vertical bits are furniture sides, not walls


def walls(V, Nv, g):
    vert = V[(np.abs(Nv[:, 2]) < 0.3) & (V[:, 2] > 0.1) & (V[:, 2] < 2.4)]
    rc = g.cell(vert)
    zmin, zmax = np.full(g.shape, 9.0), np.full(g.shape, -9.0)
    np.minimum.at(zmin, rc, vert[:, 2])
    np.maximum.at(zmax, rc, vert[:, 2])
    face = (zmax - zmin > 1.0).astype(np.uint8)
    low = g.zeros()
    low[g.cell(vert[vert[:, 2] < 0.6])] = 1
    solid = cv.dilate(cv.morphologyEx(face, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((3, 3), np.uint8))
    # glass between wall pieces: Hough finds the wall line through collinear pieces (frames, mullions, a board) across
    # gaps up to 3 m, at least 1.5 m long overall, which a fridge or a shelf can't make. Axis-aligned only (the frame is).
    sealed = face.copy()
    found = cv.HoughLinesP(face * 255, 1, np.pi / 2, threshold=20, minLineLength=30, maxLineGap=60)
    for x0, y0, x1, y1 in ([] if found is None else found.reshape(-1, 4)):
        cv.line(sealed, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    sealed = cv.dilate(sealed, np.ones((3, 3), np.uint8))
    return face, low, lines(face), solid, sealed


def runs(v, close=0):
    """Index runs [a, b] of the true cells of v, joining runs at most `close` cells apart."""
    out = []
    for i in np.flatnonzero(v):
        if out and i - out[-1][1] <= close + 1:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return out


def lines(face):
    """Straight walls: per column (axis 0) and row (axis 1), the runs of face cells in a 3-cell band; the strongest
    position wins and takes the runs of its neighbours within 6 cells (both faces of a thick wall are one wall)."""
    out = []
    for ax in (0, 1):
        f = face if ax == 0 else face.T
        band = f.copy()
        band[:, 1:] |= f[:, :-1]
        band[:, :-1] |= f[:, 1:]
        at = [[r for r in runs(band[:, k], 3) if r[1] - r[0] + 1 >= MIN_RUN] for k in range(f.shape[1])]
        score = np.array([sum(r[1] - r[0] + 1 for r in rs) for rs in at])
        taken = []
        for k in np.argsort(-score):
            if score[k] < MIN_RUN:
                break
            if any(abs(k - t) <= 6 for t in taken):
                continue
            taken.append(k)
            iv = sorted(r for j in range(max(0, k - 6), min(f.shape[1], k + 7)) for r in at[j])
            merged = [list(iv[0])]
            for a, b in iv[1:]:
                if a - merged[-1][1] <= 10:  # < 50 cm apart: the same wall
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            out.append((ax, int(k), merged))
    return out


def walls_from(edges, doors):
    """Outline edges → walls. Facing edges of two spaces < 35 cm apart become one shared wall as thick as the gap, cut
    where a door is; the rest are outer walls, their inner face on the outline."""
    segs = []
    for name, a, b in edges:
        ax = 0 if abs(a[0] - b[0]) < 1e-6 else 1
        pos, lo, hi = (a[0], min(a[1], b[1]), max(a[1], b[1])) if ax == 0 else (a[1], min(a[0], b[0]), max(a[0], b[0]))
        segs.append([name, ax, pos, lo, hi])
    walls = []
    for i, (na, ax, pa, la, ha) in enumerate(segs):
        for j, (nb, bx, pb, lb, hb) in enumerate(segs):
            if j > i and na != nb and ax == bx and 0 < abs(pa - pb) < 0.35 and min(ha, hb) - max(la, lb) > 0.3:
                lo, hi, pos, t = max(la, lb), min(ha, hb), (pa + pb) / 2, abs(pa - pb)
                cuts = [(d["center"][1 - ax] - d["width"] / 2, d["center"][1 - ax] + d["width"] / 2) for d in doors
                        if d["axis"] == ax and abs(d["center"][ax] - pos) < DOOR_REACH]
                walls += [(ax, pos, a, b, t, True) for a, b in subtract(lo, hi, cuts)]
                segs[i][3:] = [la, max(la, lo)] if la < lo else [min(hi, ha), ha]  # what is left of each edge stays outer
                segs[j][3:] = [lb, max(lb, lo)] if lb < lo else [min(hi, hb), hb]
    for name, ax, pos, lo, hi in segs:  # a door on two faces further apart than a shared wall still opens both
        cuts = [(d["center"][1 - ax] - d["width"] / 2, d["center"][1 - ax] + d["width"] / 2) for d in doors
                if d["axis"] == ax and abs(d["center"][ax] - pos) < DOOR_REACH]
        walls += [(ax, pos, a, b, WALL_T, False) for a, b in subtract(lo, hi, cuts) if hi - lo > 0.1]

    def in_door(w):  # outline stubs left inside a doorway by the cut (10 cm pieces across it)
        p, q = seg(w)
        return any(all(abs(pt[1 - d["axis"]] - d["center"][1 - d["axis"]]) <= d["width"] / 2 + 0.05 and
                       abs(pt[d["axis"]] - d["center"][d["axis"]]) <= DOOR_REACH for pt in (p, q)) for d in doors)
    return [w for w in walls if not in_door(w)]


def seg(w):
    ax, pos, lo, hi = w[:4]
    return ([pos, lo], [pos, hi]) if ax == 0 else ([lo, pos], [hi, pos])


def subtract(lo, hi, cuts):
    out = []
    for a, b in sorted(cuts):
        if a > lo:
            out.append((lo, min(a, hi)))
        lo = max(lo, b)
    return [(a, b) for a, b in out + [(lo, hi)] if b - a > 0.1]
