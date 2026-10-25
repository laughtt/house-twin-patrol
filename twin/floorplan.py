"""Floor plan of any home from a LiDAR walk (OpenCV), in four stages (twin/plan/): floor → walls → doors → rooms.

Everything is in the Manhattan-aligned, floor-at-zero frame of lidar/pipeline/build_house.py. Walls come back as
(axis, position, lo, hi, thickness, shared) tuples: axis 0 = a wall along y at x = position, axis 1 = along x at y = position.
With `out`, each stage leaves a picture there: stage_walls.png, stage_doors.png, stage_rooms.png.
"""
import cv2 as cv
import numpy as np

from twin.plan import CELL, Grid  # noqa: F401 - CELL: build_house's report
from twin.plan.doors import doors as find_doors
from twin.plan.floor import floor
from twin.plan.rooms import rooms
from twin.plan.walls import walls, walls_from


def plan(V, Nv, traj, tlab, out=None):
    g = Grid(V, traj)
    seen, path = floor(V, Nv, traj, g)
    face, low, wall_lines, solid, sealed = walls(V, Nv, g)
    found, seals = find_doors(wall_lines, face, low, seen, path, g)
    spaces, zones, edges, doors = rooms(seen, path, solid, sealed, face, found, seals, g, traj, tlab)
    if out:
        pictures(out, seen, path, face, wall_lines, found, seals, spaces, g)
    return spaces, zones, walls_from(edges, doors), doors


def pictures(out, seen, path, face, wall_lines, found, seals, spaces, g, px=3):
    base = np.full(g.shape + (3,), 255, np.uint8)
    base[seen > 0] = (228, 228, 228)
    base[path > 0] = (250, 210, 170)
    base[face > 0] = (110, 110, 110)
    up = lambda img: cv.resize(img[::-1], None, fx=px, fy=px, interpolation=cv.INTER_NEAREST)  # noqa: E731
    line = lambda img, ax, pos, a, b, col, t: cv.line(img, *(((pos, a), (pos, b)) if ax == 0 else ((a, pos), (b, pos))), col, t)  # noqa: E731
    img = base.copy()
    for ax, pos, rs in wall_lines:
        for a, b in rs:
            line(img, ax, pos, a, b, (40, 40, 220), 1)
    cv.imwrite(str(out / "stage_walls.png"), up(img))
    for ax, pos, a, b in seals:
        line(img, ax, pos, a, b, (0, 140, 255), 2)
    for d in found:
        line(img, d["axis"], *d["cells"], (60, 170, 60) if d["walked"] else (0, 200, 255), 3)
    cv.imwrite(str(out / "stage_doors.png"), up(img))
    img = base.copy()
    rng = np.random.default_rng(1)
    for s in spaces:
        img[s["mask"] > 0] = rng.integers(130, 235, 3)
    img[face > 0] = (110, 110, 110)
    img = up(img)
    for s in spaces:
        ys, xs = np.nonzero(s["mask"][::-1])
        cv.putText(img, s["name"].replace("_", " "), (int(xs.mean() * px) - 40, int(ys.mean() * px)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv.LINE_AA)
    cv.imwrite(str(out / "stage_rooms.png"), img)
