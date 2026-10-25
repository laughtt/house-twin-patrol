"""The map the duck localizes in: what a sensor 20 cm above the floor can hit, its distance map, a fast ray cast, and
the spin match's library of range profiles."""
import math

import cv2 as cv
import numpy as np

from agent.loc import CELL, TOF_MAX


class HouseMap:
    """What a sensor 20 cm above the floor can hit: walls, solid furniture, and only the legs of tables and chairs."""

    def __init__(self, house, eye_h=0.2, occupancy=None):
        xs = [p for w in house["walls"] for p in (w["a"][0], w["b"][0])]
        ys = [p for w in house["walls"] for p in (w["a"][1], w["b"][1])]
        self.o = np.minimum(0.0, [min(xs), min(ys)])  # grid origin: (0, 0), or lower if a room lies west or south of it
        self.W, self.L = max(xs) - self.o[0], max(ys) - self.o[1]
        self.grid = np.zeros((int(self.L / CELL) + 3, int(self.W / CELL) + 3), bool)
        self.grid[:, :1] = self.grid[:, -2:] = self.grid[:1, :] = self.grid[-2:, :] = True
        for w in house["walls"]:  # every wall, interior ones too, as thick as it is
            a, b = (((np.array(w[k]) - self.o) / CELL + 1).astype(int) for k in ("a", "b"))
            self.grid |= cv.line(np.zeros(self.grid.shape, np.uint8), tuple(int(v) for v in a), tuple(int(v) for v in b), 1,
                                 max(1, int(round(w["t"] / CELL)))).astype(bool)
        for o in house["objects"]:
            (cx, cy), (w, d, h), yaw = o["center"], o["size"], o["yaw"]
            legs = o["kind"] == "table" or o["label"] == "chair"
            if h < eye_h - 0.06 and not legs:  # the ToF rows look ±3° off level: from ~1 m they hit things 6 cm lower
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            parts = [(sx * (w / 2 - 0.04), sy * (d / 2 - 0.04), 0.02, 0.02) for sx in (-1, 1) for sy in (-1, 1)] if legs else [(0, 0, w / 2, d / 2)]
            for px, py, hw, hd in parts:
                u, v = np.meshgrid(np.arange(-hw, hw + CELL, CELL / 2), np.arange(-hd, hd + CELL, CELL / 2))
                x, y = cx + (px + u) * c - (py + v) * s - self.o[0], cy + (px + u) * s + (py + v) * c - self.o[1]
                ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.L)
                self.grid[(y[ok] / CELL).astype(int) + 1, (x[ok] / CELL).astype(int) + 1] = True
        if occupancy:  # the scan's real surfaces at the sensor's height (house.json "occupancy"), beyond walls and boxes
            z = np.load(occupancy)
            band = z["occ"][max(int((eye_h - 0.1) / z["dz"]), 0):int((eye_h + 0.1) / z["dz"]) + 1].any(0)
            band = cv.morphologyEx(band.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)  # 5 cm gaps
            yy, xx = np.mgrid[0:self.grid.shape[0], 0:self.grid.shape[1]]
            i = (((yy - 1) * CELL + self.o[1] - z["lo"][1]) / z["cell"]).astype(int)
            j = (((xx - 1) * CELL + self.o[0] - z["lo"][0]) / z["cell"]).astype(int)
            ok = (i >= 0) & (j >= 0) & (i < band.shape[0]) & (j < band.shape[1])
            self.grid[ok] |= band[i[ok], j[ok]]
        self.markers = {m["id"]: m for m in house.get("markers", [])}
        self.landmarks = [lm for lm in house.get("landmarks", []) if lm.get("cls") and lm.get("weight", 0) > 0]
        self.rooms = house.get("rooms", [])
        self.room_of = np.full(self.grid.shape, -1, np.int16)  # which room each cell is in (-1: outside every room)
        for k, r in enumerate(self.rooms):
            for x0, y0, x1, y1 in r["rects"]:
                (x0, y0), (x1, y1) = np.subtract([x0, y0], self.o), np.subtract([x1, y1], self.o)
                self.room_of[int(y0 / CELL) + 1:int(y1 / CELL) + 2, int(x0 / CELL) + 1:int(x1 / CELL) + 2] = k
        self.clear = cv.distanceTransform((~self.grid).astype(np.uint8), cv.DIST_L2, 5) * CELL  # metres to the nearest hit

    def room(self, x, y):
        i = np.clip(((np.asarray(y) - self.o[1]) / CELL).astype(int) + 1, 0, self.grid.shape[0] - 1)
        j = np.clip(((np.asarray(x) - self.o[0]) / CELL).astype(int) + 1, 0, self.grid.shape[1] - 1)
        return self.room_of[i, j]

    def free(self, x, y):
        i, j = ((np.asarray(y) - self.o[1]) / CELL).astype(int) + 1, ((np.asarray(x) - self.o[0]) / CELL).astype(int) + 1
        inside = (i >= 0) & (i < self.grid.shape[0]) & (j >= 0) & (j < self.grid.shape[1])
        return inside & ~self.grid[np.clip(i, 0, self.grid.shape[0] - 1), np.clip(j, 0, self.grid.shape[1] - 1)]

    def raycast(self, x, y, angle, step=0.02, rmax=TOF_MAX):
        """Distance to the first hit for many rays at once (rmax + 1 when nothing within range). Same samples as a
        2 cm march from 5 cm out (to rmax, the sensor's range), but each ray jumps over the ones that the distance map says can't hit anything."""
        x, y, angle = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, angle)))
        shape, x, y = x.shape, x.ravel(), y.ravel()
        c, s = np.cos(angle).ravel(), np.sin(angle).ravel()
        dist, k, idx = np.full(x.size, rmax + 1.0), np.zeros(x.size, np.int64), np.arange(x.size)
        last = int(round((rmax - 0.05) / step))
        while idx.size:
            r = 0.05 + k[idx] * step
            i = ((y[idx] + r * s[idx] - self.o[1]) / CELL).astype(int) + 1
            j = ((x[idx] + r * c[idx] - self.o[0]) / CELL).astype(int) + 1
            inside = (i >= 0) & (i < self.grid.shape[0]) & (j >= 0) & (j < self.grid.shape[1])
            d = np.where(inside, self.clear[i.clip(0, self.grid.shape[0] - 1), j.clip(0, self.grid.shape[1] - 1)], 0.0)
            hit = d == 0
            dist[idx[hit]] = r[hit]
            k[idx] += np.maximum(1, ((0.98 * d - 1.5 * CELL) / step).astype(np.int64))  # a point this far from any obstacle cell
            idx = idx[~hit & (k[idx] <= last)]
        return dist.reshape(shape)

    def profiles(self, rmax, step=0.10, bins=120):
        """The range every 3° from every free 10 cm spot (in a room, where the duck's body fits): the spin match's
        library, with each spot's lattice index (row, col). Once per house and range."""
        cache = self.__dict__.setdefault("prof", {})
        if rmax not in cache:
            gy, gx = np.mgrid[0:int(self.L / step) + 1, 0:int(self.W / step) + 1]
            xs, ys = self.o[0] + gx.ravel() * step, self.o[1] + gy.ravel() * step
            i, j = ((ys - self.o[1]) / CELL).astype(int) + 1, ((xs - self.o[0]) / CELL).astype(int) + 1
            keep = (self.room(xs, ys) >= 0) & (self.clear[i.clip(0, self.grid.shape[0] - 1), j.clip(0, self.grid.shape[1] - 1)] > 0.1)
            xs, ys = xs[keep], ys[keep]
            cache[rmax] = (np.c_[xs, ys], np.c_[gy.ravel(), gx.ravel()][keep],
                           self.raycast(xs[:, None], ys[:, None], (np.arange(bins) * 2 * np.pi / bins)[None], rmax=rmax).astype(np.float32))
        return cache[rmax]
