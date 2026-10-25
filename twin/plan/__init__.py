"""Floor plan of a home from a LiDAR walk, one stage per module: floor → walls → doors → rooms (twin/floorplan.py runs
them). All stages share one 5 cm grid in the Manhattan-aligned, floor-at-zero frame of lidar/pipeline/build_house.py.
Walls and doors use axis 0 = along y at x = position, axis 1 = along x at y = position."""
import numpy as np

CELL = 0.05


class Grid:
    def __init__(self, V, traj):
        self.lo = np.minimum(V[:, :2].min(0), traj[:, :2].min(0)) - 0.5
        self.shape = tuple((np.ceil((np.maximum(V[:, :2].max(0), traj[:, :2].max(0)) + 0.5 - self.lo) / CELL)).astype(int)[::-1] + 1)

    def cell(self, p):  # → (rows, cols)
        return tuple(((np.asarray(p)[..., :2] - self.lo) / CELL).astype(int).T[::-1])

    def to_m(self, r, c):
        return [float(self.lo[0] + c * CELL), float(self.lo[1] + r * CELL)]

    def zeros(self):
        return np.zeros(self.shape, np.uint8)
