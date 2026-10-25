"""Stage 4, rooms: the floor cut by the walls, with every door drawn shut, falls apart into the rooms. An area is a room
when it has 2 m² and the walk went in, or a door joins it to one (a room only looked into). Zone taps only name the
rooms (most of the walk points inside); two tapped names that each hold a good share of one room's walk make an
open-plan space of two zones. A room no tap names is called "Room N" here and named from what is in it later."""
from collections import deque

import cv2 as cv
import numpy as np

from twin.plan import CELL
from twin.plan.doors import close

MIN_ROOM_M2 = 2.0
CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
ZONE_SHARE = 0.15  # a tapped name needs this share of a room's walk points to be a zone of it (taps come a little late)
NAMED_BY = [("Bathroom", {"bathtub", "toilet", "shower"}), ("Bedroom", {"bed"}), ("Kitchen", {"fridge", "refrigerator", "oven", "stove"}),
            ("Office", {"desk"}), ("Living room", {"sofa", "couch", "TV", "tv"})]


def rooms(seen, path, solid, sealed, face, found, seals, g, traj, tlab):
    wide = filled((seen & (1 - sealed)) | path)
    walk = filled((seen & (1 - (sealed | solid))) | (path & (1 - solid)))  # the walk reopens glass and bridged gaps, never a wall it saw
    cut = walk.copy()
    for s in seals:
        close(cut, *s, pad=2)
    for d in found:
        close(cut, d["axis"], *d["cells"], pad=3)
    n, comp = cv.connectedComponents(cut, connectivity=4)
    area = np.bincount(comp.ravel(), minlength=n) * CELL ** 2
    rc = [g.cell(p) for p in traj]
    inside = [int(comp[r, c]) for r, c in rc]
    keep = {k for k in inside if k > 0 and area[k] >= MIN_ROOM_M2}

    def sides(d, big):  # the first big area off each side of a door, within 1 m of its line
        pos, lo, hi = d["cells"]
        mid, out = (lo + hi) // 2, []
        for s in (-1, 1):
            hits = [int(comp[p]) for p in ((mid, pos + s * k) if d["axis"] == 0 else (pos + s * k, mid) for k in range(4, 21))
                    if 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]]
            out.append(next((h for h in hits if h in big), 0))
        return out
    big = {k for k in range(1, n) if area[k] >= MIN_ROOM_M2}
    for d in found:
        a, b = sides(d, big)
        for x, y in ((a, b), (b, a)):
            if x in keep and y:
                keep.add(y)
    doors = [{k: v for k, v in d.items() if k != "cells"} for d in found if len(set(sides(d, keep)) - {0}) == 2]

    votes = {k: {} for k in keep}
    for k, name in zip(inside, tlab):
        if k in keep:
            votes[k][name] = votes[k].get(name, 0) + 1
    order = sorted(keep, key=lambda k: inside.index(k) if k in inside else len(inside) + k)  # in the order the walk went in
    # a name goes to the room where it was tapped most; rooms left without one are named from their contents later
    owner = {nm: max(order, key=lambda k: votes[k].get(nm, 0)) for k in keep for nm in votes[k]}
    members = {k: [nm for nm, v in sorted(votes[k].items(), key=lambda kv: -kv[1])
                   if owner[nm] == k and v >= ZONE_SHARE * sum(votes[k].values())] for k in order}
    # each room then takes back the floor `solid` cut off (furniture joined to a wall), by nearest room, never through a
    # wall it saw: the outlines are those of the plain floor, only split where walls and doors split it
    grown = np.where(np.isin(comp, list(keep)), comp, 0).astype(np.float32)
    while True:
        more = np.where((grown == 0) & (wide > 0), cv.dilate(grown, CROSS), grown)  # 4-connected: not through a wall's corner
        if (more == grown).all():
            break
        grown = more
    comp = grown.astype(np.int32)
    spaces, zones, edges, unnamed = [], [], [], 0
    for k in order:
        names = members[k]
        if not names:
            unnamed += 1
            names = [f"Room {unnamed}"]
        mask = (comp == k).astype(np.uint8)
        # close notches up to 75 cm: tall furniture against a wall (a fridge) hides the floor and looks like wall.
        # A square kernel keeps true 90° corners (an L stays an L).
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        poly = rectilinear(max(cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0], key=cv.contourArea), mask, face)
        name = "+".join(slug(nm) for nm in names)
        pts = [g.to_m(r, c) for c, r in poly]
        spaces.append({"name": name, "polygon": pts, "mask": mask, "lo": g.lo, "cell": CELL})
        edges += [(name, pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        if len(names) == 1:  # a room of its own: its zone is the whole wall-to-wall outline
            m = cv.fillPoly(g.zeros(), [poly.astype(np.int32)], 1)
            zones.append({"name": slug(names[0]), "label": names[0], "space": name, "polygon": pts,
                          "rects": [g.to_m(r0, c0) + g.to_m(r1, c1) for r0, c0, r1, c1 in rects_of(m.astype(bool), CELL, 0.02)]})
            continue
        # open plan: each cell of the room takes the name of the nearest walk point tapped with one of its names
        zone = np.full(g.shape, -1, np.int32)
        q = deque()
        for (r, c), kk, nm in zip(rc, inside, tlab):
            if kk == k and nm in names and zone[r, c] < 0:
                zone[r, c] = names.index(nm)
                q.append((r, c))
        while q:
            r, c = q.popleft()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nb[0] < g.shape[0] and 0 <= nb[1] < g.shape[1] and comp[nb] == k and zone[nb] < 0:
                    zone[nb] = zone[r, c]
                    q.append(nb)
        for i, nm in enumerate(names):
            m = (zone == i).astype(np.uint8)
            if m.any():
                c = max(cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)
                zones.append({"name": slug(nm), "label": nm, "space": name,
                              "polygon": [g.to_m(r, c_) for c_, r in cv.approxPolyDP(c, 1.0, True)[:, 0]],
                              "rects": [g.to_m(r0, c0) + g.to_m(r1, c1) for r0, c0, r1, c1 in rects_of(m.astype(bool), CELL, 0.02)]})
    return spaces, zones, edges, doors


def filled(m):
    """Unseen pockets enclosed by walkable cells (under a table, behind a chair) are walkable too."""
    holes = m.copy()
    cv.floodFill(holes, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return m | (holes == 0).astype(np.uint8)


def name_rooms(spaces, zones, landmarks):
    """"Room N" → what its landmarks say it is (a bed: Bedroom), unless that name is taken."""
    taken = {z["label"] for z in zones}
    for z in zones:
        if not z["label"].startswith("Room "):
            continue
        inside = {lm["label"] for lm in landmarks for r in z["rects"] if r[0] <= lm["center"][0] <= r[2] and r[1] <= lm["center"][1] <= r[3]}
        for nm, what in NAMED_BY:
            if inside & what:
                new = nm if nm not in taken else next(f"{nm} {i}" for i in range(2, 99) if f"{nm} {i}" not in taken)
                taken.add(new)
                for s in spaces:
                    if s["name"] == z["space"]:
                        s["name"] = slug(new)
                z.update(name=slug(new), label=new, space=slug(new))
                break


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
        probe = (lambda d: inside[min(H - 1, max(0, v + d)), min(W - 1, mid)]) if kind == "h" else (lambda d: inside[min(H - 1, mid), min(W - 1, max(0, v + d))])
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
