"""Stage 3, doors: openings in the wall lines. From each free end of a wall run (not a T or a corner: a wall running
across it would end there) walk along its line to the next wall, in line or across. 0.55-1.3 m with floor seen on both
sides and nothing low across it is a door; a narrower gap is a bit of wall the scan missed, an opening blocked low down
is a window (wall below the glass) or furniture in the way: both sealed. Wider is open plan. The walk is not needed:
`walked` only says whether the person went through."""

from twin.plan import CELL

DOOR = (0.55, 1.3)
WIDTH_FIX = 1.5  # cells: the jambs' faces are seen a little inside the opening (19 test homes: 10 cm narrow)


def doors(wall_lines, face, low, seen, path, g):
    found, seals = [], []
    for ax, pos, rs in wall_lines:
        f = face if ax == 0 else face.T  # f[along, across]
        band = f[:, max(0, pos - 2):pos + 3].any(1)
        for a, b in rs:
            for end, step in ((b, 1), (a, -1)):
                across = [f[min(len(band) - 1, max(0, end + e)), pos + d] for e in (-1, 0, 1) for d in (-7, -6, -5, 5, 6, 7)
                          if 0 <= pos + d < f.shape[1]]
                if sum(across) >= 2:
                    continue
                n = 1
                while 0 <= end + step * n < len(band) and not band[end + step * n] and n <= 27:
                    n += 1
                if n == 1 or not 0 <= end + step * n < len(band) or (n - 1) * CELL > DOOR[1]:
                    continue
                lo, hi = sorted((end + step, end + step * (n - 1)))
                sill = (low[lo:hi + 1, max(0, pos - 2):pos + 3].any(1) if ax == 0 else low[max(0, pos - 2):pos + 3, lo:hi + 1].any(0)).mean()
                if (n - 1) * CELL < DOOR[0] or sill >= 0.5:
                    seals.append((ax, pos, lo, hi))
                    continue
                mid = (lo + hi) // 2
                both = all(any(seen[(mid, pos + s * d) if ax == 0 else (pos + s * d, mid)]
                               for d in range(2, 13) if 0 <= pos + s * d < f.shape[1]) for s in (-1, 1))
                if not both or any(d["axis"] == ax and abs(d["cells"][0] - pos) <= 8 and d["cells"][1] <= mid <= d["cells"][2] for d in found):
                    continue
                walked = bool((path[lo:hi + 1, pos] if ax == 0 else path[pos, lo:hi + 1]).any())
                center = g.to_m(mid + 0.5, pos + 0.5) if ax == 0 else g.to_m(pos + 0.5, mid + 0.5)
                found.append({"axis": ax, "center": center, "width": round((hi - lo + 1 + WIDTH_FIX) * CELL, 3), "walked": walked,
                              "cells": (pos, lo, hi)})
    return found, seals


def close(grid, ax, pos, lo, hi, pad):
    """Draw an opening shut across (pad cells each side of the line, and past both jambs)."""
    if ax == 0:
        grid[max(0, lo - pad):hi + pad + 1, max(0, pos - pad):pos + pad + 1] = 0
    else:
        grid[max(0, pos - pad):pos + pad + 1, max(0, lo - pad):hi + pad + 1] = 0
