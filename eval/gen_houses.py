"""Varied synthetic homes with truth, written like the iPhone app's export (renderer: eval/synth_lidar.py).

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/gen_houses.py [name ...]
→ out/synth_houses/<name>.zip + <name>.truth.json (score them with eval/score_many.py).

A house = rooms as rectangles on a 5 cm grid. Walls come from the room map: 12 cm centred on the border between two
rooms (unless the pair is open plan), 20 cm outside the outer border; doors and glass are cut out of them. The walk is
planned on the free floor (30 cm from walls and furniture) through a list of targets, with a 360° turn at each target
and, in some rooms, a level duck-height turn. Truth polygons run along the wall centre lines.
"""
import json
import shutil
import sys
import zipfile
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import synth_lidar as S  # noqa: E402

OUT = Path(__file__).parent.parent / "out" / "synth_houses"
G, INNER, OUTER, CLEAR, STEP, TURN, EYE = 0.05, 0.12, 0.2, 0.3, 0.2, 24, 1.35
DETECTABLE = {"couch", "bed", "dining table", "refrigerator", "tv", "toilet", "potted plant", "chair", "oven", "sink", "counter"}


def tv(x0, y0, x1, y):  # a TV on a low cabinet against a wall at y (cabinet footprint x0..x1, y0..y)
    return [("cabinet", x0, y0, x1, y, 0.5), ("tv", x0 + 0.15, (y0 + y) / 2 - 0.04, x1 - 0.15, (y0 + y) / 2 + 0.04, 1.15, 0.5)]


def chairs(x0, y0, x1, y1):  # four chairs around a table
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, 0.22
    return [("chair", x0 - 0.5, cy - s, x0 - 0.05, cy + s, 0.9), ("chair", x1 + 0.05, cy - s, x1 + 0.5, cy + s, 0.9),
            ("chair", cx - s, y0 - 0.5, cx + s, y0 - 0.05, 0.9), ("chair", cx - s, y1 + 0.05, cx + s, y1 + 0.5, 0.9)]


# furniture: (label, x0, y0, x1, y1, height[, bottom]); heights are above the room's floor
HOUSES = [
    dict(name="h01_studio", note="1 room, low density, single zone", yaw=0.3, ceiling=2.5,
         rooms={"Studio": [(0, 0, 5, 4)]},
         furniture=[("couch", 0.2, 3.1, 2.2, 3.9, 0.85), ("dining table", 3.0, 1.0, 4.2, 1.8, 0.75)],
         visits=[("Studio", 1.3, 1.3), ("Studio", 3.8, 2.8)], duck={"Studio": 0.45}),
    dict(name="h02_two_rooms", note="2 walled rooms, door 0.9", yaw=0.7, ceiling=2.5,
         rooms={"Living room": [(0, 0, 4.5, 4)], "Bedroom": [(4.5, 0, 8, 4)]}, doors=[(4.5, 2.0, 0.9)],
         furniture=[("couch", 0.2, 3.1, 2.4, 3.9, 0.85), *tv(1.0, 0.05, 2.4, 0.5), ("bed", 5.8, 2.0, 7.8, 3.9, 0.55),
                    ("wardrobe", 4.7, 0.05, 5.9, 0.65, 2.0)],
         visits=["Living room", "Bedroom"], duck={"Living room": 0.45}),
    dict(name="h03_l_open_glass", note="open-plan L (kitchen+living), glass wall 3 m, medium density", yaw=1.1, ceiling=2.6,
         rooms={"Kitchen": [(0, 0, 3.5, 3.5)], "Living room": [(0, 3.5, 7, 7)]}, open=[("Kitchen", "Living room")],
         glass=[(3.5, 7.0, 3.0)],
         furniture=[("counter", 0.05, 0.6, 0.65, 2.9, 0.9), ("refrigerator", 2.7, 0.05, 3.4, 0.75, 1.8),
                    ("dining table", 1.0, 4.4, 2.2, 5.2, 0.75), ("couch", 4.5, 5.9, 6.8, 6.8, 0.85),
                    ("potted plant", 6.3, 3.8, 6.7, 4.2, 0.9)],
         visits=["Kitchen", "Living room", ("Living room", 5.5, 4.8)]),
    dict(name="h04_corridor5", note="corridor + 4 rooms off it, doors 0.8/1.0", yaw=2.0, ceiling=2.5,
         rooms={"Hallway": [(0, 3, 8, 4.2)], "Bathroom": [(0, 0, 2.5, 3)], "Bedroom": [(2.5, 0, 5.5, 3)],
                "Office": [(5.5, 0, 8, 3)], "Living room": [(0, 4.2, 8, 8)]},
         doors=[(1.25, 3, 0.8), (4.0, 3, 0.8), (6.75, 3, 0.8), (4.0, 4.2, 1.0)],
         furniture=[("toilet", 0.15, 0.2, 0.55, 0.8, 0.45), ("sink", 1.6, 0.05, 2.3, 0.5, 0.85),
                    ("bed", 3.0, 0.1, 5.0, 2.0, 0.55), ("desk", 6.0, 0.05, 7.6, 0.7, 0.75), ("chair", 6.5, 0.8, 6.95, 1.25, 0.9),
                    ("couch", 0.3, 7.0, 2.5, 7.9, 0.85), ("dining table", 5.0, 5.5, 6.4, 6.4, 0.75), *tv(0.6, 4.3, 2.2, 4.75)],
         visits=["Living room", ("Hallway", 2.5, 3.6), "Bathroom", "Bedroom", "Office"],
         duck={"Living room": 0.5, "Bedroom": 0.4}),
    dict(name="h05_six_rooms", note="6 rooms, hall, kitchen+dining open, doors 0.7-1.0, high density", yaw=2.6, ceiling=2.6,
         rooms={"Hallway": [(3, 0, 4.5, 8)], "Kitchen": [(0, 0, 3, 4)], "Dining room": [(0, 4, 3, 8)],
                "Bedroom": [(4.5, 0, 8, 3)], "Bathroom": [(4.5, 3, 8, 5)], "Office": [(4.5, 5, 8, 8)]},
         open=[("Kitchen", "Dining room")],
         doors=[(3, 2, 1.0), (3, 6, 0.9), (4.5, 1.5, 0.8), (4.5, 4, 0.7), (4.5, 6.5, 0.9)],
         furniture=[("counter", 0.05, 0.5, 0.65, 3.5, 0.9), ("refrigerator", 2.2, 0.05, 2.9, 0.7, 1.8),
                    ("counter", 1.3, 1.7, 2.0, 2.7, 0.9), ("dining table", 0.9, 5.5, 2.1, 6.5, 0.75), *chairs(0.9, 5.5, 2.1, 6.5),
                    ("bed", 6.0, 0.1, 7.9, 2.2, 0.55), ("wardrobe", 4.65, 2.3, 5.8, 2.9, 2.0),
                    ("potted plant", 7.4, 2.4, 7.85, 2.85, 0.9), ("toilet", 7.3, 3.2, 7.9, 3.6, 0.45),
                    ("bathtub", 5.2, 4.2, 7.9, 4.9, 0.55), ("sink", 6.0, 3.05, 6.8, 3.5, 0.85),
                    ("desk", 6.0, 7.3, 7.9, 7.95, 0.75), ("chair", 6.7, 6.8, 7.15, 7.25, 0.9),
                    ("bookcase", 4.6, 7.5, 5.6, 7.95, 1.8)],
         visits=["Kitchen", "Dining room", ("Hallway", 3.75, 4), "Bedroom", "Bathroom", "Office"],
         duck={"Dining room": 0.45, "Office": 0.35}),
    dict(name="h06_single_zone", note="4 walled rooms, one zone label for the whole walk", yaw=0.9, ceiling=2.5, taps="one",
         rooms={"Living room": [(0, 0, 5, 4)], "Kitchen": [(5, 0, 8, 4)], "Bedroom": [(0, 4, 4, 7.5)], "Bathroom": [(4, 4, 8, 7.5)]},
         doors=[(5, 2, 0.9), (2, 4, 0.8), (4, 6, 0.8)],
         furniture=[("couch", 0.2, 0.2, 2.4, 1.0, 0.85), ("counter", 7.35, 0.4, 7.95, 3.6, 0.9),
                    ("refrigerator", 5.2, 3.25, 5.9, 3.95, 1.8), ("bed", 0.5, 5.5, 2.5, 7.4, 0.55),
                    ("toilet", 7.4, 6.8, 7.9, 7.4, 0.45), ("sink", 5.0, 7.0, 5.8, 7.45, 0.85)],
         visits=["Living room", "Kitchen", "Bedroom", "Bathroom"]),
    dict(name="h07_missed_tap", note="kitchen, hall, bedroom in a row; the hallway tap was forgotten", yaw=1.5, ceiling=2.5,
         rooms={"Kitchen": [(0, 0, 4, 3.5)], "Hallway": [(4, 0, 5.4, 3.5)], "Bedroom": [(5.4, 0, 9, 3.5)]},
         doors=[(4, 1.75, 0.85), (5.4, 1.75, 0.85)], missed=["Hallway"],
         furniture=[("counter", 0.05, 0.5, 0.65, 3.0, 0.9), ("refrigerator", 3.2, 2.75, 3.9, 3.45, 1.8),
                    ("bed", 7.0, 0.5, 8.95, 2.5, 0.55)],
         visits=["Kitchen", "Hallway", "Bedroom"]),
    dict(name="h08_sunken_living", note="kitchen + sunken living (-13 cm) open plan, bedroom behind a door", yaw=0.2, ceiling=2.6,
         rooms={"Kitchen": [(0, 0, 4, 4)], "Living room": [(4, 0, 9, 5)], "Bedroom": [(0, 4, 4, 7.5)]},
         open=[("Kitchen", "Living room")], doors=[(2, 4, 0.9)], sunken={"Living room": -0.13},
         furniture=[("counter", 0.05, 0.5, 0.65, 3.5, 0.9), ("refrigerator", 2.5, 0.05, 3.2, 0.7, 1.8),
                    ("couch", 5.0, 3.95, 7.5, 4.85, 0.85), *tv(5.5, 0.05, 7.0, 0.5), ("bed", 0.5, 5.5, 2.5, 7.4, 0.55)],
         visits=["Kitchen", "Living room", ("Living room", 7.8, 2.2), "Bedroom"], duck={"Living room": 0.45}),
    dict(name="h09_threshold", note="raised 4 cm threshold at the bathroom door, irregular outline", yaw=1.8, ceiling=2.5,
         rooms={"Living room": [(0, 0, 5, 4.5)], "Bathroom": [(5, 0, 7.5, 2.5)], "Bedroom": [(5, 2.5, 8.5, 6)]},
         doors=[(5, 1.25, 0.8), (5, 3.5, 0.85)], thresholds=[(5, 1.25, 0.8)],
         furniture=[("toilet", 6.9, 0.2, 7.4, 0.8, 0.45), ("sink", 5.6, 0.05, 6.4, 0.5, 0.85),
                    ("couch", 0.3, 3.6, 2.5, 4.4, 0.85), ("dining table", 1.5, 1.0, 2.7, 1.8, 0.75),
                    ("bed", 6.5, 4.0, 8.4, 5.9, 0.55)],
         visits=["Living room", "Bathroom", "Bedroom"]),
    dict(name="h10_unvisited", note="storage room the walk never enters (seen through its door)", yaw=2.3, ceiling=2.5,
         rooms={"Living room": [(0, 0, 6, 4)], "Bedroom": [(0, 4, 3.5, 7)], "Storage": [(3.5, 4, 6, 7)]},
         doors=[(1.75, 4, 0.9), (4.75, 4, 0.8)],
         furniture=[("bookcase", 3.6, 6.4, 5.9, 6.95, 1.8), ("bed", 0.2, 5.0, 2.2, 6.9, 0.55),
                    ("couch", 0.2, 0.2, 2.4, 1.0, 0.85), ("dining table", 3.8, 1.2, 5.0, 2.0, 0.75)],
         visits=["Living room", "Bedroom", ("Living room", 4.75, 3.0)]),
    dict(name="h11_narrow_doors", note="3 rooms in a row, 0.7 m doors, 2.4 m ceiling, low density", yaw=3.0, ceiling=2.4,
         rooms={"Living room": [(0, 0, 4, 3.5)], "Bedroom": [(4, 0, 7.5, 3.5)], "Bathroom": [(7.5, 0, 9.5, 3.5)]},
         doors=[(4, 1.75, 0.7), (7.5, 1.0, 0.7)],
         furniture=[("couch", 0.2, 2.6, 2.4, 3.4, 0.85), ("bed", 5.0, 1.5, 7.0, 3.4, 0.55), ("toilet", 8.9, 2.8, 9.4, 3.4, 0.45)],
         visits=["Living room", "Bedroom", "Bathroom"]),
    dict(name="h12_wide_doors", note="1.0 m doors, irregular outline, 2.7 m ceiling", yaw=0.5, ceiling=2.7,
         rooms={"Kitchen": [(0, 0, 3.5, 4)], "Living room": [(3.5, 0, 8, 4.5)], "Study": [(3.5, 4.5, 8, 7)]},
         doors=[(3.5, 2, 1.0), (5.75, 4.5, 1.0)],
         furniture=[("counter", 0.05, 0.5, 0.65, 3.5, 0.9), ("refrigerator", 2.7, 3.25, 3.4, 3.95, 1.8),
                    ("couch", 4.5, 0.2, 6.7, 1.0, 0.85), ("desk", 6.0, 6.3, 7.9, 6.95, 0.75), ("chair", 6.7, 5.8, 7.15, 6.25, 0.9)],
         visits=["Kitchen", "Living room", "Study"]),
    dict(name="h13_loft", note="one 8x6 open space, 3 zones, 3.0 m ceiling, glass, high density", yaw=1.3, ceiling=3.0,
         rooms={"Kitchen": [(0, 0, 4, 6)], "Dining room": [(4, 0, 8, 3)], "Living room": [(4, 3, 8, 6)]},
         open=[("Kitchen", "Dining room"), ("Kitchen", "Living room"), ("Dining room", "Living room")], glass=[(8, 4.5, 2.0)],
         furniture=[("counter", 0.05, 0.5, 0.65, 5.0, 0.9), ("counter", 1.6, 2.0, 2.6, 4.0, 0.9),
                    ("refrigerator", 0.05, 5.2, 0.75, 5.9, 1.8), ("dining table", 5.4, 1.0, 6.6, 1.8, 0.75),
                    *chairs(5.4, 1.0, 6.6, 1.8), ("couch", 5.0, 5.1, 7.4, 5.95, 0.85), ("potted plant", 7.4, 3.2, 7.85, 3.65, 0.9),
                    ("potted plant", 3.4, 5.4, 3.85, 5.85, 0.9)],
         visits=["Kitchen", "Dining room", "Living room"], duck={"Dining room": 0.45}),
    dict(name="h14_low_ceiling", note="2.3 m ceiling, 2 rooms, low density, yaw 0", yaw=0.0, ceiling=2.3,
         rooms={"Living room": [(0, 0, 4.5, 3.5)], "Bedroom": [(4.5, 0, 7.5, 3.5)]}, doors=[(4.5, 1.2, 0.8)],
         furniture=[("couch", 0.2, 2.6, 2.4, 3.4, 0.85), ("bed", 5.4, 1.4, 7.4, 3.4, 0.55)],
         visits=["Living room", "Bedroom"]),
    dict(name="h15_u_shape", note="U-shaped open plan, 3 zones", yaw=2.8, ceiling=2.5,
         rooms={"Living room": [(0, 0, 2.5, 6)], "Dining room": [(2.5, 0, 5.5, 2.5)], "Kitchen": [(5.5, 0, 8, 6)]},
         open=[("Living room", "Dining room"), ("Dining room", "Kitchen")],
         furniture=[("couch", 0.1, 3.5, 0.95, 5.8, 0.85), ("dining table", 3.4, 0.8, 4.6, 1.7, 0.75),
                    ("counter", 7.35, 1.0, 7.95, 5.0, 0.9), ("refrigerator", 6.2, 5.25, 6.9, 5.95, 1.8)],
         visits=[("Living room", 1.4, 4.5), "Dining room", ("Kitchen", 6.6, 4.0)]),
    dict(name="h16_t_glass", note="T-shaped outline, 3 m glass on the living room's south wall", yaw=0.6, ceiling=2.6,
         rooms={"Living room": [(0, 2, 9, 6)], "Bedroom": [(3, 6, 6, 9.5)]}, doors=[(4.5, 6, 0.9)], glass=[(4.5, 2, 3.0)],
         furniture=[("couch", 1.0, 4.8, 3.2, 5.7, 0.85), ("dining table", 6.0, 3.2, 7.2, 4.0, 0.75),
                    ("potted plant", 8.4, 5.4, 8.85, 5.85, 0.9), ("bed", 3.5, 7.5, 5.5, 9.4, 0.55)],
         visits=[("Living room", 2.0, 3.4), ("Living room", 7.5, 4.8), "Bedroom"], duck={"Living room": 0.5}),
    dict(name="h17_dense_flat", note="3 rooms, very dense furniture", yaw=1.0, ceiling=2.5,
         rooms={"Living room": [(0, 0, 5, 4)], "Kitchen": [(5, 0, 7.5, 4)], "Bedroom": [(0, 4, 4, 7)]},
         doors=[(5, 3.0, 0.8), (2, 4, 0.8)],
         furniture=[("couch", 0.1, 0.1, 2.3, 0.95, 0.85), ("couch", 0.1, 1.3, 0.95, 2.2, 0.85),
                    ("table", 1.3, 1.5, 2.3, 2.1, 0.45), *tv(2.8, 3.45, 4.4, 3.9), ("potted plant", 4.5, 0.1, 4.9, 0.5, 0.9),
                    ("potted plant", 0.1, 3.4, 0.5, 3.8, 0.9), ("bookcase", 0.6, 3.45, 1.4, 3.9, 1.8),
                    ("dining table", 3.2, 1.0, 4.2, 1.8, 0.75), ("chair", 3.45, 0.5, 3.9, 0.95, 0.9),
                    ("counter", 6.85, 0.1, 7.45, 3.9, 0.9), ("counter", 5.1, 0.05, 6.4, 0.65, 0.9),
                    ("refrigerator", 6.1, 3.3, 6.8, 3.95, 1.8),
                    ("bed", 0.1, 5.0, 2.0, 6.95, 0.55), ("wardrobe", 2.5, 6.35, 3.95, 6.95, 2.0), ("desk", 3.3, 4.6, 3.95, 5.9, 0.75),
                    ("chair", 2.8, 5.0, 3.25, 5.45, 0.9), ("potted plant", 0.1, 4.1, 0.5, 4.5, 0.9)],
         visits=["Living room", "Kitchen", "Bedroom"]),
    dict(name="h18_duck_heavy", note="duck-height turns in every room (0.3 / 0.45 / 0.6 m)", yaw=2.1, ceiling=2.5,
         rooms={"Living room": [(0, 0, 5, 4.5)], "Kitchen": [(5, 0, 8, 4.5)], "Bedroom": [(0, 4.5, 5, 7.5)]},
         open=[("Living room", "Kitchen")], doors=[(2.5, 4.5, 0.9)],
         furniture=[("couch", 0.2, 0.2, 2.4, 1.0, 0.85), ("counter", 7.35, 0.5, 7.95, 4.0, 0.9),
                    ("refrigerator", 5.2, 0.05, 5.9, 0.75, 1.8), ("bed", 0.5, 5.5, 2.5, 7.4, 0.55)],
         visits=["Living room", "Kitchen", "Bedroom"], duck={"Living room": 0.3, "Kitchen": 0.45, "Bedroom": 0.6}),
    dict(name="h19_sparse_walk", note="h02's rooms, sparse walk (16-frame turns, 25 cm steps)", yaw=0.7, ceiling=2.5, turn=16, step=0.25,
         rooms={"Living room": [(0, 0, 4.5, 4)], "Bedroom": [(4.5, 0, 8, 4)]}, doors=[(4.5, 2.0, 0.9)],
         furniture=[("couch", 0.2, 3.1, 2.4, 3.9, 0.85), *tv(1.0, 0.05, 2.4, 0.5), ("bed", 5.8, 2.0, 7.8, 3.9, 0.55),
                    ("wardrobe", 4.7, 0.05, 5.9, 0.65, 2.0)],
         visits=["Living room", "Bedroom"], duck={"Living room": 0.45}),
]


def box3(ax, across, along, z):
    return ((across[0], along[0], z[0]), (across[1], along[1], z[1])) if ax == 0 else \
        ((along[0], across[0], z[0]), (along[1], across[1], z[1]))


def runs(m):
    d = np.diff(np.r_[0, m.astype(int), 0])
    return zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])


def poly_of(mask, lo):
    """Cell mask → its exact outline (cell corners), metres."""
    c = max(cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)[:, 0].astype(float)
    s = np.sign(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
    out = []
    for k in range(len(c)):
        e1, e2 = c[k] - c[k - 1], c[(k + 1) % len(c)] - c[k]
        n1, n2 = (s * np.array([e[1], -e[0]]) / np.linalg.norm(e) for e in (e1, e2))
        out.append((lo + (c[k] + 0.5 + 0.5 * (n1 + n2)) * G).round(3).tolist())
    return out


class Plan:
    def __init__(self, h):
        self.h, self.names = h, list(h["rooms"])
        rects = [r for v in h["rooms"].values() for r in v]
        self.lo = np.min([r[:2] for r in rects], 0) - 4.0  # room for outer walls and a patio behind glass
        hi = np.max([r[2:] for r in rects], 0) + 4.0
        self.R = np.full(tuple(np.round((hi - self.lo) / G).astype(int)[::-1]), -1, int)
        for k, v in enumerate(h["rooms"].values()):
            for x0, y0, x1, y1 in v:
                (c0, r0), (c1, r1) = self.cell((x0, y0)), self.cell((x1, y1))
                self.R[r0:r1, c0:c1] = k
        self.fz = [h.get("sunken", {}).get(n, 0.0) for n in self.names]

    def cell(self, p):
        return tuple(np.round((np.asarray(p, float) - self.lo) / G).astype(int))  # (col, row) of the corner / cell

    def at(self, p):
        c, r = np.floor((np.asarray(p, float) - self.lo) / G).astype(int)
        return self.R[r, c]

    def opening(self, x, y, w):
        """A door or glass section centred on a border at (x, y): (axis, position, span lo, span hi)."""
        return (0, x, y - w / 2, y + w / 2) if self.at((x - 0.1, y)) != self.at((x + 0.1, y)) else (1, y, x - w / 2, x + w / 2)

    def solids(self):
        h, ceil = self.h, self.h["ceiling"]
        n = len(self.names)
        OP = np.zeros((n + 1, n + 1), bool)
        for a, b in h.get("open", []):
            OP[self.names.index(a) + 1, self.names.index(b) + 1] = OP[self.names.index(b) + 1, self.names.index(a) + 1] = True
        doors = [self.opening(*d) for d in h.get("doors", [])]
        glass = [self.opening(*g) for g in h.get("glass", [])]
        walls = []
        for ax in (0, 1):  # ax 0: borders between columns = walls along y at x
            A = self.R if ax == 0 else self.R.T
            la, lc = self.lo[1 - ax], self.lo[ax]
            a, b = A[:, :-1], A[:, 1:]
            border = (a != b) & ~OP[a + 1, b + 1]
            kind = np.where((a >= 0) & (b >= 0), 0, np.where(a < 0, 1, 2))
            cut = np.zeros_like(border)
            mids = la + (np.arange(A.shape[0]) + 0.5) * G
            for cax, pos, s0, s1 in doors + glass:
                if cax == ax:
                    cut[(mids > s0) & (mids < s1), int(round((pos - lc) / G)) - 1] = True
            for j in range(border.shape[1]):
                x = lc + (j + 1) * G
                for k, across, e in ((0, (x - INNER / 2, x + INNER / 2), INNER / 2), (1, (x - OUTER, x), OUTER), (2, (x, x + OUTER), OUTER)):
                    side = j if k == 1 else j + 1  # the outside cells an outer wall stands on
                    for i0, i1 in runs(border[:, j] & (kind[:, j] == k) & ~cut[:, j]):
                        ext = lambda i: 0 <= i < len(mids) and not cut[i, j] and (k == 0 or A[i, side] < 0)  # noqa: E731
                        walls.append(box3(ax, across, (la + i0 * G - e * ext(i0 - 1), la + i1 * G + e * ext(i1)), (-0.3, ceil)))
        out = [((*self.lo, ceil), (*(self.lo + np.array(self.R.shape[::-1]) * G), ceil + 0.2))]  # ceiling
        for k, v in enumerate(h["rooms"].values()):
            out += [((x0, y0, -0.3), (x1, y1, self.fz[k])) for x0, y0, x1, y1 in v]
        out += walls
        for ax, pos, s0, s1 in doors:
            out.append(box3(ax, (pos - INNER / 2, pos + INNER / 2), (s0, s1), (2.05, ceil)))
        for ax, pos, s0, s1 in [self.opening(*t) for t in h.get("thresholds", [])]:
            out.append(box3(ax, (pos - 0.1, pos + 0.1), (s0, s1), (-0.3, 0.04)))
        panes = []
        for ax, pos, s0, s1 in glass:  # pane on the outer face, header above, patio floor and a fence outside
            p = np.array([0.0, 0.0])
            p[ax], p[1 - ax] = pos + 0.1, (s0 + s1) / 2
            o = 1 if self.at(p) < 0 else -1
            out.append(box3(ax, sorted((pos, pos + o * OUTER)), (s0, s1), (2.2, ceil)))
            out.append(box3(ax, sorted((pos + o * OUTER, pos + o * 3.5)), (s0 - 1, s1 + 1), (-0.3, -0.02)))
            out.append(box3(ax, sorted((pos + o * 3.5, pos + o * 3.55)), (s0 - 1, s1 + 1), (-0.3, 1.2)))
            panes.append(len(out))
            out.append(box3(ax, sorted((pos, pos + o * 0.04)), (s0, s1), (-0.3, 2.2)))
        return out, panes, doors, glass

    def furniture(self):
        out = []
        for f in self.h.get("furniture", []):
            label, x0, y0, x1, y1, ht = f[:6]
            z = self.fz[self.at(((x0 + x1) / 2, (y0 + y1) / 2))]
            out.append((label, (x0, y0, z + (f[6] if len(f) > 6 else 0.0)), (x1, y1, z + ht)))
        return out

    def free(self, solids, furn):
        """Floor the person can stand on: inside a room, 30 cm clear of walls and furniture."""
        occ = (self.R < 0).astype(np.uint8)
        yy, xx = (np.mgrid[0:self.R.shape[0], 0:self.R.shape[1]] + 0.5) * G
        xx, yy = xx + self.lo[0], yy + self.lo[1]
        for lo, hi in [s for s in solids if s[1][2] - max(s[0][2], 0) > 0.5 and s[0][2] < 1.5] + [(a, b) for _, a, b in furn]:
            occ[(xx > lo[0]) & (xx < hi[0]) & (yy > lo[1]) & (yy < hi[1])] = 1
        free = (cv.distanceTransform(1 - occ, cv.DIST_L2, 5) * G >= CLEAR).astype(np.uint8)
        lab = cv.connectedComponents(free, connectivity=8)[1]
        return lab == np.bincount(lab[free > 0]).argmax()  # pockets between furniture are not reachable


def bfs(free, a, b):
    prev = {a: None}
    q = deque([a])
    while q:
        p = q.popleft()
        if p == b:
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (p[0] + dr, p[1] + dc)
                if n not in prev and 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1] and free[n]:
                    prev[n] = p
                    q.append(n)
    if b not in prev:
        raise ValueError(f"no free path from {a} to {b}")
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def pull(path, free):
    """Shortest path's cells → straight segments where the floor between is free."""
    def los(p, q):
        t = np.linspace(0, 1, int(np.hypot(q[0] - p[0], q[1] - p[1]) * 2) + 2)
        r, c = np.round(p[0] + (q[0] - p[0]) * t).astype(int), np.round(p[1] + (q[1] - p[1]) * t).astype(int)
        return free[r, c].all()
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not los(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def walk(plan, free):
    """→ per frame: xy, yaw, pitch, camera height above the local floor."""
    h = plan.h
    rc = lambda p: tuple(np.floor((np.asarray(p) - plan.lo) / G).astype(int)[::-1])  # noqa: E731
    xy = lambda c: plan.lo + (np.array(c[::-1]) + 0.5) * G  # noqa: E731
    targets = []
    for v in h["visits"]:
        name, p = (v, None) if isinstance(v, str) else (v[0], v[1:])
        if p is None:  # the middle of the room's biggest rectangle
            x0, y0, x1, y1 = max(h["rooms"][name], key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            p = ((x0 + x1) / 2, (y0 + y1) / 2)
        cand = np.argwhere(free & (plan.R == plan.names.index(name)))
        targets.append((name, tuple(cand[np.argmin(np.hypot(*(cand - rc(p)).T))])))
    frames, ducked = [], set()
    def add(p, yaw, pitch=np.radians(15), ht=EYE):
        frames.append((np.asarray(p, float), float(yaw), float(pitch), float(ht)))
    yaw = 0.0
    for k, (name, t) in enumerate(targets):
        if k:
            way = [xy(c) for c in pull(bfs(free, targets[k - 1][1], t), free)]
            for a, b in zip(way[:-1], way[1:]):
                n = max(1, int(np.ceil(np.hypot(*(b - a)) / h.get("step", STEP))))
                yaw = np.arctan2(*(b - a)[::-1])
                for s in np.arange(n) / n:
                    add(a + (b - a) * s, yaw + np.radians(55) * np.sin(len(frames) * 0.35))
        p = xy(t)
        turn = h.get("turn", TURN)
        for j in range(1, turn + 1):
            add(p, yaw + j * 2 * np.pi / turn)
        if name in h.get("duck", {}) and name not in ducked:  # down to duck height, a level turn, back up
            ducked.add(name)
            low = h["duck"][name]
            for j in range(1, 4):
                add(p, yaw + j * 0.3, np.radians(15) * (1 - j / 3), EYE + (low - EYE) * j / 3)
            for j in range(1, turn + 1):
                add(p, yaw + 0.9 + j * 2 * np.pi / turn, 0.0, low)
            add(p, yaw + 0.9, np.radians(8), (EYE + low) / 2)
        yaw += 2 * np.pi
    return frames


def label_frames(plan, frames):
    """The zone the person taps: a step inside a room, until the next one (taps "one": a single zone)."""
    h, out, prev = plan.h, [], None
    for p, *_ in frames:
        room = next((n for n, v in h["rooms"].items() if any(x0 + 0.3 <= p[0] <= x1 - 0.3 and y0 + 0.3 <= p[1] <= y1 - 0.3
                                                           for x0, y0, x1, y1 in v)), None)
        if room and room not in h.get("missed", []):
            prev = room
        out.append("Home" if h.get("taps") == "one" else prev or plan.names[0])
    return out


def box_px(R, c, lo, hi, sid, k, f, wh):
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pc = (pts - c) @ R
    if (pc[:, 2] <= 0.2).any():
        return None
    uv = pc[:, :2] / pc[:, 2:] * f + np.array(wh) / 2
    x0, y0 = np.maximum(uv.min(0), 0)
    x1, y1 = np.minimum(uv.max(0), wh)
    if x1 - x0 < 10 or y1 - y0 < 10 or (sid[int(y0):int(y1), int(x0):int(x1)] == k).mean() < 0.3:
        return None
    return [float(v) * S.RGB_WH[0] / wh[0] for v in (x0, y0, x1, y1)]


def generate(h):
    plan = Plan(h)
    solids, panes, doors, glass = plan.solids()
    furn = plan.furniture()
    S.BOXES, S.GLASS, S.MARKERS = solids + [(a, b) for _, a, b in furn], set(panes), []
    frames = walk(plan, plan.free(solids, furn))
    labels = label_frames(plan, frames)
    rng = np.random.default_rng(sum(map(ord, h["name"])))
    RZ = np.array([[np.cos(h["yaw"]), -np.sin(h["yaw"]), 0], [np.sin(h["yaw"]), np.cos(h["yaw"]), 0], [0, 0, 1]])
    root = OUT / "_tmp" / h["name"]
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("rgb", "depth", "conf"):
        (root / sub).mkdir(parents=True)
    half = (S.RGB_WH[0] // 2, S.RGB_WH[1] // 2)
    fd = S.F * S.DEPTH_WH[0] / S.RGB_WH[0]
    K = lambda s: [S.F * s, S.F * s, S.RGB_WH[0] / 2 * s, S.RGB_WH[1] / 2 * s]  # noqa: E731
    lines, events, boxes2d, t0 = [], [], {}, None
    for i, ((p, yaw, pitch, ht), room) in enumerate(zip(frames, labels)):
        R, c = S.pose(p, yaw, pitch)
        c = np.array([p[0], p[1], plan.fz[plan.at(p)] + ht])
        if t0 is None:
            t0 = -RZ @ c
        name = f"{i:06d}"
        img, _, _, sid = S.render(R, c, half, S.F / 2)
        img = cv.resize(img, S.RGB_WH, interpolation=cv.INTER_LINEAR)
        cv.imwrite(str(root / f"rgb/{name}.jpg"), cv.GaussianBlur(img, (0, 0), 0.7), [cv.IMWRITE_JPEG_QUALITY, 60])
        _, z, cos, _ = S.render(R, c, S.DEPTH_WH, fd, glass=False)
        z = z * (1 + 0.004 * rng.standard_normal(z.shape)) + 0.003 * rng.standard_normal(z.shape)
        conf = np.where(z > 5, 0, np.where((z < 3.5) & (cos > 0.3), 2, 1)).astype(np.uint8)
        (root / f"depth/{name}.bin").write_bytes(np.minimum(z, 9).astype(np.float16).tobytes())
        (root / f"conf/{name}.bin").write_bytes(conf.tobytes())
        jitter = cv.Rodrigues(rng.normal(0, np.radians(0.15), 3))[0]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = S.M.T @ RZ @ jitter @ R @ np.diag([1, -1, -1]), S.M.T @ (RZ @ c + t0) + rng.normal(0, 0.004, 3)
        lines.append({"i": i, "t": i / 4, "rgb": f"rgb/{name}.jpg", "depth": f"depth/{name}.bin", "conf": f"conf/{name}.bin",
                      "rgb_wh": list(S.RGB_WH), "K": K(1), "depth_wh": list(S.DEPTH_WH), "K_depth": K(S.DEPTH_WH[0] / S.RGB_WH[0]),
                      "T": T.ravel().tolist(), "tracking": "normal", "room": room, "h": round(ht, 2)})
        if i and labels[i - 1] != room:
            events.append({"t": i / 4, "i": i, "event": "room", "room": room, "T": T.ravel().tolist()})
        boxes2d[name] = [(lab, b) for k, (lab, lo, hi) in enumerate(furn)
                         if lab in DETECTABLE and (b := box_px(R, c, lo, hi, sid, len(solids) + k, S.F / 2, half))]
    (root / "frames.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (root / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in events))
    (root / "session.json").write_text(json.dumps({"session": h["name"], "device": "synthetic", "fps": 4, "format": 2}))
    zpath = OUT / f"{h['name']}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            z.write(f, f.relative_to(root.parent))
    shutil.rmtree(root)

    # truth
    parent = list(range(len(plan.names)))
    find = lambda i: i if parent[i] == i else find(parent[i])  # noqa: E731
    for a, b in h.get("open", []):
        parent[find(plan.names.index(a))] = find(plan.names.index(b))
    spaces = {}
    for i, n in enumerate(plan.names):
        spaces.setdefault(find(i), []).append(n)
    walked = {plan.at(p) for p, *_ in frames}
    zone_of = {}
    for p, lab in zip(frames, labels):
        zone_of.setdefault(plan.names[plan.at(p[0])], {}).setdefault(lab, 0)
        zone_of[plan.names[plan.at(p[0])]][lab] += 1
    truth = {"name": h["name"], "note": h["note"],
             "frame": "z-up metres; house_to_world maps it to the z-up world of build_house.Export.load poses",
             "house_to_world": np.r_[np.c_[RZ, t0], [[0, 0, 0, 1]]].round(6).tolist(),
             "ceiling": h["ceiling"], "frames": len(frames), "frames_low": int(sum(f[3] < 0.75 for f in frames)),
             "zones": sorted(set(labels)),
             "rooms": [{"name": n, "polygon": poly_of(plan.R == k, plan.lo), "rects": h["rooms"][n], "floor_z": plan.fz[k],
                        "visited": k in walked, "zone": max(zone_of[n], key=zone_of[n].get) if n in zone_of else None}
                       for k, n in enumerate(plan.names)],
             "outer_polygon": poly_of(plan.R >= 0, plan.lo),
             "spaces": list(spaces.values()),
             "doors": [{"center": [pos, (s0 + s1) / 2] if ax == 0 else [(s0 + s1) / 2, pos], "width": round(s1 - s0, 3), "axis": ax}
                       for ax, pos, s0, s1 in doors],
             "thresholds": [list(t) for t in h.get("thresholds", [])],
             "glass": [{"a": [pos, s0] if ax == 0 else [s0, pos], "b": [pos, s1] if ax == 0 else [s1, pos], "h": 2.2}
                       for ax, pos, s0, s1 in glass],
             "furniture": [{"label": lab, "lo": list(lo), "hi": list(hi), "room": plan.names[plan.at(np.add(lo, hi)[:2] / 2)]}
                           for lab, lo, hi in furn],
             "landmarks": [{"cls": lab, "center": (np.add(lo, hi) / 2).round(3).tolist(), "size": np.subtract(hi, lo).round(3).tolist(),
                            "room": plan.names[plan.at(np.add(lo, hi)[:2] / 2)]} for lab, lo, hi in furn if lab in DETECTABLE],
             "boxes2d": boxes2d}
    (OUT / f"{h['name']}.truth.json").write_text(json.dumps(truth, indent=1))
    return zpath, len(frames)


if __name__ == "__main__":
    import time
    OUT.mkdir(parents=True, exist_ok=True)
    want = set(sys.argv[1:])
    for h in HOUSES:
        if want and h["name"] not in want:
            continue
        tic = time.time()
        zpath, n = generate(h)
        print(f"{h['name']:20s} {n:4d} frames  {zpath.stat().st_size / 1e6:5.1f} MB  {time.time() - tic:5.1f} s  {h['note']}", flush=True)
