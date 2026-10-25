"""Stage 1, floor: what LiDAR saw as horizontal and low (the floor and furniture tops: a counter, a bed, a sofa seat), and
the footprint of the walk (15 cm around each step). Walls, glass and the outside give no such surface."""
import cv2 as cv

from twin.plan import CELL


def floor(V, Nv, traj, g):
    flat = (Nv[:, 2] > 0.8) & (V[:, 2] > -0.06) & (V[:, 2] < 1.2)
    seen = g.zeros()
    seen[g.cell(V[flat])] = 1
    path = g.zeros()
    for p in traj:
        cv.circle(path, (int((p[0] - g.lo[0]) / CELL), int((p[1] - g.lo[1]) / CELL)), 3, 1, -1)
    return seen, path
