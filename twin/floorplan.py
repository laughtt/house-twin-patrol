"""Floor plan of any home from a LiDAR walk (OpenCV): zones grown along the floor from where the person walked.

Everything is in the Manhattan-aligned, floor-at-zero frame of lidar/pipeline/build_house.py. Walls come back as
(axis, position, lo, hi, thickness) tuples: axis 0 = a wall along y at x = position, axis 1 = along x at y = position.
"""
import os
from collections import deque

import cv2 as cv
import numpy as np

CELL = 0.05
OPEN_BORDER = 1.2  # zones touching along more than this (m) are one open space; less is a doorway
WALL_T = 0.10


def plan(V, Nv, traj, tlab):
    lo = np.minimum(V[:, :2].min(0), traj[:, :2].min(0)) - 0.5
    shape = tuple((np.ceil((np.maximum(V[:, :2].max(0), traj[:, :2].max(0)) + 0.5 - lo) / CELL)).astype(int)[::-1] + 1)
    cell = lambda p: tuple(((np.asarray(p)[..., :2] - lo) / CELL).astype(int).T[::-1])  # noqa: E731 → (rows, cols)
    to_m = lambda r, c: [float(lo[0] + c * CELL), float(lo[1] + r * CELL)]  # noqa: E731

    # walkable = what LiDAR saw as horizontal and low: the floor and furniture tops (a counter, a bed, a sofa seat),
    # plus where the person stood. Walls, glass and the outside give no such surface, so they bound the zones.
    flat = (Nv[:, 2] > 0.8) & (V[:, 2] > -0.06) & (V[:, 2] < 1.2)
    floor = np.zeros(shape, np.uint8)
    floor[cell(V[flat])] = 1
    # walls: vertical surfaces spanning at least 1 m of height in a cell. Counter fronts (0-0.9 m), sofa backs and upper
    # cabinets (1.4-2.2 m) are not walls; a door's lintel isn't either, so doorways stay open. Floor seen on both sides
    # of a thin wall would otherwise leak through it.
    vert = V[(np.abs(Nv[:, 2]) < 0.3) & (V[:, 2] > 0.1) & (V[:, 2] < 2.4)]
    rc = cell(vert)
    zmin, zmax = np.full(shape, 9.0), np.full(shape, -9.0)
    np.minimum.at(zmin, rc, vert[:, 2])
    np.maximum.at(zmax, rc, vert[:, 2])
    wall_face = (zmax - zmin > 1.0).astype(np.uint8)
    wall = cv.dilate(wall_face, np.ones((3, 3), np.uint8))
    # glass doors and windows between wall sections: LiDAR sees the patio floor through them. Gaps up to 1.5 m in a
    # straight wall line are sealed; where the person actually walked is opened again, so real doorways stay open.
    path = np.zeros(shape, np.uint8)
    for p in traj:
        cv.circle(path, (int((p[0] - lo[0]) / CELL), int((p[1] - lo[1]) / CELL)), 3, 1, -1)
    # glass between wall pieces: Hough finds the wall line through collinear pieces (frames, mullions, a board) across
    # gaps up to 3 m, at least 1.5 m long overall, which a fridge or a shelf can't make. Axis-aligned only (the frame is).
    sealed = wall_face.copy()
    lines = cv.HoughLinesP(wall_face * 255, 1, np.pi / 2, threshold=20, minLineLength=30, maxLineGap=60)
    for x0, y0, x1, y1 in ([] if lines is None else lines.reshape(-1, 4)):
        cv.line(sealed, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    sealed = cv.dilate(sealed, np.ones((3, 3), np.uint8))
    walk = (floor & (1 - sealed)) | path
    open_walk = (floor & (1 - wall)) | path  # unsealed: for measuring doorways
    holes = walk.copy()  # unseen pockets enclosed by walkable cells (under a table, behind a chair) are walkable too
    cv.floodFill(holes, np.zeros((shape[0] + 2, shape[1] + 2), np.uint8), (0, 0), 1)
    walk |= (holes == 0).astype(np.uint8)

    # at each zone tap, look back along the walk for the narrowest place squeezed through: < 1.3 m is a doorway
    # (separate rooms), wider is open plan. Doorways are then cut into the walkable map so no zone leaks through.
    names = list(dict.fromkeys(tlab))
    dist = cv.distanceTransform(walk, cv.DIST_L2, 5)
    if os.environ.get("FLOORPLAN_DEBUG"):
        dbg = cv.cvtColor(walk[::-1] * 200, cv.COLOR_GRAY2BGR)
        dbg[(sealed & (1 - wall))[::-1] > 0] = (0, 160, 255)  # sealed gaps in orange
        dbg[wall_face[::-1] > 0] = (90, 90, 90)
        for p in traj:
            rc = cell(p)
            dbg[shape[0] - 1 - rc[0], rc[1]] = (0, 0, 255)
        cv.imwrite(os.environ["FLOORPLAN_DEBUG"], cv.resize(dbg, None, fx=3, fy=3, interpolation=cv.INTER_NEAREST))
    parent = list(range(len(names)))
    root = lambda i: i if parent[i] == i else root(parent[i])  # noqa: E731
    doors = []
    for k in range(1, len(traj)):
        a, b = names.index(tlab[k - 1]), names.index(tlab[k])
        if a == b:
            continue
        def opening(i):  # (cells across the walk that are walkable, both sides) at walk point i
            r, c = cell(traj[i])
            step = traj[i + 1, :2] - traj[i - 1, :2]
            axis = 0 if abs(step[0]) > abs(step[1]) else 1  # walking along x crosses a wall that runs along y
            dr, dc = (1, 0) if axis == 0 else (0, 1)
            ends, jambs = [], 0
            for s in (-1, 1):
                n = 0
                while n < 60 and 0 <= r + s * dr * (n + 1) < shape[0] and 0 <= c + s * dc * (n + 1) < shape[1] and open_walk[r + s * dr * (n + 1), c + s * dc * (n + 1)]:
                    n += 1
                ends.append(n)
                rr, cc = r + s * dr * (n + 1), c + s * dc * (n + 1)
                jambs += 0 <= rr < shape[0] and 0 <= cc < shape[1] and bool(wall[rr, cc])
            # a doorway is an opening with a wall at both ends; furniture or unseen floor beside the path is not
            return (sum(ends) if jambs == 2 else 999), ends, axis, r, c, dr, dc
        j = min(range(max(1, k - 15), min(len(traj) - 1, k + 3)), key=lambda i: opening(i)[0])
        n_open, ends, axis, r, c, dr, dc = opening(j)
        if n_open == 999:  # no walls either side anywhere near the tap: open plan
            parent[root(a)] = root(b)
            continue
        width = (sum(ends) + 3) * CELL  # + the one-cell wall margin on each side
        if os.environ.get("FLOORPLAN_DEBUG"):
            print(f"tap {tlab[k - 1]} -> {tlab[k]} at frame {k}: narrowest frame {j}, half-width {dist[r, c] * CELL:.2f} m, "
                  f"run across {width:.2f} m (axis {axis})")
        if width >= 1.3:
            parent[root(a)] = root(b)
            continue
        mid = (ends[1] - ends[0]) / 2
        center = to_m(r + mid * dr + 0.5, c + mid * dc + 0.5)
        if all(np.hypot(*np.subtract(center, d["center"])) > 0.5 for d in doors):
            doors.append({"axis": axis, "center": center, "width": round(float(width), 3), "zones": (a, b)})
    for d in doors:  # cut: one cell across the doorway, a little wider than it
        (r, c), half = cell(d["center"]), int(d["width"] / 2 / CELL) + 2
        if d["axis"] == 0:
            walk[max(0, r - half):r + half + 1, c] = 0
        else:
            walk[r, max(0, c - half):c + half + 1] = 0

    # each walkable cell takes the zone of the nearest walked point, counted along the walkable cells (not through walls)
    zone = np.full(shape, -1, np.int32)
    q = deque()
    for p, name in zip(traj, tlab):
        rc = cell(p)
        if walk[rc] and zone[rc] < 0:
            zone[rc] = names.index(name)
            q.append(rc)
    while q:
        r, c = q.popleft()
        for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and walk[n] and zone[n] < 0:
                zone[n] = zone[r, c]
                q.append(n)
    # zones never walked between directly but sharing a long border are open plan too
    for x, y in ((zone[:, :-1], zone[:, 1:]), (zone[:-1, :], zone[1:, :])):
        m = (x >= 0) & (y >= 0) & (x != y)
        pairs, counts = np.unique(np.sort(np.c_[x[m], y[m]], 1), axis=0, return_counts=True) if m.any() else ([], [])
        for (i, j), n in zip(pairs, counts):
            if n * CELL > 2 * OPEN_BORDER and not any(set(d["zones"]) == {i, j} for d in doors):
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(names)):
        groups.setdefault(root(i), []).append(i)

    spaces, zones, edges = [], [], []
    for members in groups.values():
        mask = np.isin(zone, members).astype(np.uint8)
        if not mask.any():
            continue
        # close notches up to 75 cm: tall furniture against a wall (a fridge) hides the floor and looks like wall.
        # A square kernel keeps true 90° corners (an L stays an L).
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        poly = rectilinear(max(cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0], key=cv.contourArea), mask, wall_face)
        name = "+".join(slug(names[i]) for i in members)
        pts = [to_m(r, c) for c, r in poly]
        spaces.append({"name": name, "polygon": pts, "mask": mask, "lo": lo, "cell": CELL})
        edges += [(name, pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
        if len(members) == 1:  # a room of its own: its zone is the whole wall-to-wall outline
            m = cv.fillPoly(np.zeros(shape, np.uint8), [poly.astype(np.int32)], 1)
            zones.append({"name": slug(names[members[0]]), "label": names[members[0]], "space": name, "polygon": pts,
                          "rects": [to_m(r0, c0) + to_m(r1, c1) for r0, c0, r1, c1 in rects_of(m.astype(bool), CELL, 0.02)]})
            continue
        for i in members:
            m = (zone == i).astype(np.uint8)
            if m.any():
                c = max(cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)
                zones.append({"name": slug(names[i]), "label": names[i], "space": name,
                              "polygon": [to_m(r, c_) for c_, r in cv.approxPolyDP(c, 1.0, True)[:, 0]],
                              "rects": [to_m(r0, c0) + to_m(r1, c1) for r0, c0, r1, c1 in rects_of(m.astype(bool), CELL, 0.02)]})
    doors = [{k: v for k, v in d.items() if k != "zones"} for d in doors]
    return spaces, zones, walls_from(edges, doors), doors


def rectilinear(contour, inside, wall, reach=12):
    """Contour → polygon with only horizontal and vertical edges (the frame is already turned to the walls). Each edge is
    then pushed out to the wall behind it, if one is within `reach` cells: the floor strip along a wall is rarely seen.
    Where nothing is behind (glass, the open side of a doorway), the edge stays where the floor ends."""
    poly = cv.approxPolyDP(contour, 2.0, True)[:, 0].astype(float)
    out = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        out.append(("h", (a[1] + b[1]) / 2) if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else ("v", (a[0] + b[0]) / 2))
    merged = [e for k, e in enumerate(out) if e[0] != out[k - 1][0] or abs(e[1] - out[k - 1][1]) > 3]
    while len(merged) > 1 and merged[0][0] == merged[-1][0]:
        merged.pop()
    H, W = inside.shape
    for k, (kind, v) in enumerate(merged):
        a, b = sorted((merged[k - 1][1], merged[(k + 1) % len(merged)][1]))
        a, b = int(a) + 2, int(b) - 2
        if b <= a:
            continue
        v = int(round(v))
        mid = (a + b) // 2
        probe = (lambda d: inside[min(H - 1, max(0, v + d)), mid]) if kind == "h" else (lambda d: inside[mid, min(W - 1, max(0, v + d))])
        out = 1 if probe(-2) else -1  # outward = the side that is not inside
        for d in range(1, reach + 1):
            line = wall[min(H - 1, max(0, v + out * d)), a:b] if kind == "h" else wall[a:b, min(W - 1, max(0, v + out * d))]
            if line.mean() > 0.3:
                merged[k] = (kind, v + out * d)  # on the wall's face
                break
    verts = []
    for k in range(len(merged)):
        e, f = merged[k], merged[(k + 1) % len(merged)]
        if e[0] == f[0]:  # two parallel steps in a row: bridge with a short perpendicular edge at the first one's end
            continue
        verts.append((f[1], e[1]) if e[0] == "h" else (e[1], f[1]))
    out = []
    for k, v in enumerate(verts):
        out.append(v)
        w = verts[(k + 1) % len(verts)]
        if abs(w[0] - v[0]) > 1 and abs(w[1] - v[1]) > 1:  # slanted (two parallel edges met): make it a step
            out.append((w[0], v[1]))
    return np.round(out).astype(int)


def walls_from(edges, doors):
    """Outline edges → walls. Facing edges of two spaces < 35 cm apart become one shared wall as thick as the gap, cut
    where a door is; the rest are outer walls, their inner face on the outline."""
    segs = []
    for name, a, b in edges:
        ax = 0 if abs(a[0] - b[0]) < 1e-6 else 1
        pos, lo, hi = (a[0], min(a[1], b[1]), max(a[1], b[1])) if ax == 0 else (a[1], min(a[0], b[0]), max(a[0], b[0]))
        segs.append([name, ax, pos, lo, hi])
    walls, used = [], set()
    for i, (na, ax, pa, la, ha) in enumerate(segs):
        for j, (nb, bx, pb, lb, hb) in enumerate(segs):
            if j > i and na != nb and ax == bx and 0 < abs(pa - pb) < 0.35 and min(ha, hb) - max(la, lb) > 0.3:
                lo, hi, pos, t = max(la, lb), min(ha, hb), (pa + pb) / 2, abs(pa - pb)
                cuts = [(d["center"][1 - ax] - d["width"] / 2, d["center"][1 - ax] + d["width"] / 2) for d in doors
                        if d["axis"] == ax and abs(d["center"][ax] - pos) < 0.3]
                walls += [(ax, pos, a, b, t, True) for a, b in subtract(lo, hi, cuts)]
                used |= {i, j}
                segs[i][3:] = [la, max(la, lo)] if la < lo else [min(hi, ha), ha]  # what is left of each edge stays outer
                segs[j][3:] = [lb, max(lb, lo)] if lb < lo else [min(hi, hb), hb]
    for name, ax, pos, lo, hi in segs:
        if hi - lo > 0.1:
            walls.append((ax, pos, lo, hi, WALL_T, False))
    return walls


def subtract(lo, hi, cuts):
    out = []
    for a, b in sorted(cuts):
        if a > lo:
            out.append((lo, min(a, hi)))
        lo = max(lo, b)
    return [(a, b) for a, b in out + [(lo, hi)] if b - a > 0.1]


def largest_rect(m):
    best, h = (0, None), np.zeros(m.shape[1], int)
    for r in range(m.shape[0]):
        h = np.where(m[r], h + 1, 0)
        stack = []
        for c in range(len(h) + 1):
            cur, start = (h[c] if c < len(h) else 0), c
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (c - s) > best[0]:
                    best = (hh * (c - s), (r - hh + 1, s, r + 1, c))
                start = s
            stack.append((start, cur))
    return best


def rects_of(mask, cell, min_m2, min_side=0.0):
    """A shape as its largest rectangles, biggest first (row/col cells)."""
    mask, out = mask.copy(), []
    while True:
        area, rc = largest_rect(mask)
        if area * cell ** 2 < min_m2:
            return out
        r0, c0, r1, c1 = rc
        mask[r0:r1, c0:c1] = False
        if min(r1 - r0, c1 - c0) * cell >= min_side:
            out.append(rc)


def slug(s):
    return s.strip().lower().replace(" ", "_")
