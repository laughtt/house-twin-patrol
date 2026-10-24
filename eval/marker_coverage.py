"""How many wall markers does a room need? Monte Carlo coverage for the phone scan and the duck camera."""
import json
import numpy as np

rng = np.random.default_rng(0)
SIZE, MARK_H, MAX_VIEW = 0.15, 0.20, np.radians(60)
MIN_PX = 20


def cam(hfov_deg, w, h):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return f, np.arctan(w / 2 / f), np.arctan(h / 2 / f)


PHONE, DUCK = cam(66, 1920, 1080), (530.0, np.arctan(320 / 530), np.arctan(180 / 530))


def walls(W, D):  # name -> (centre xy, inward normal); order = which walls get markers first
    return {"S": ((W / 2, 0), (0, 1)), "N": ((W / 2, D), (0, -1)), "E": ((W, D / 2), (-1, 0)), "W": ((0, D / 2), (1, 0))}


def layouts(W, D):
    w = walls(W, D)
    S, N, E, Wst = w["S"], w["N"], w["E"], w["W"]
    pair = lambda wall: [((wall[0][0] + dx * abs(wall[1][1]), wall[0][1] + dx * abs(wall[1][0])), wall[1]) for dx in (-0.5, 0.5)]
    return {"1 marker": [S], "2 on opposite walls": [S, N], "2 on one wall (1 m apart)": pair(S),
            "3 on three walls": [S, N, E], "3 = pair + 1 opposite": pair(S) + [N],
            "4 on four walls": [S, N, E, Wst], "4 = two pairs, opposite walls": pair(S) + pair(N)}


def blocked(c, m, boxes):
    """Segment camera->marker hits any axis-aligned box (x0,y0,z0,x1,y1,z1)? Slab test, vectorised over cameras."""
    hit = np.zeros(len(c), bool)
    d = m - c
    for b in boxes:
        lo, hi = np.array(b[:3]), np.array(b[3:])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - c) / d, (hi - c) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit |= (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)
    return hit


def visible(c, yaw, pitch, markers, camera, boxes=()):
    """Bool matrix [n_cameras, n_markers]: marker inside FOV, faced within 60 deg, >= 20 px, not occluded."""
    f, hh, hv = camera
    fwd = np.stack([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)], 1)
    right = np.stack([np.sin(yaw), -np.cos(yaw), np.zeros_like(yaw)], 1)
    up = np.cross(right, fwd)
    out = []
    for (mx, my), (nx, ny) in markers:
        m = np.array([mx, my, MARK_H])
        v = m - c
        dist = np.linalg.norm(v, axis=1)
        z, x, y = (v * fwd).sum(1), (v * right).sum(1), (v * up).sum(1)
        cos_view = -(v[:, 0] * nx + v[:, 1] * ny) / dist
        px = f * SIZE * np.clip(cos_view, 0, 1) / dist
        ok = (z > 0) & (np.abs(np.arctan2(x, z)) <= hh) & (np.abs(np.arctan2(y, z)) <= hv)
        ok &= (cos_view >= np.cos(MAX_VIEW)) & (px >= MIN_PX) & ~blocked(c, m, boxes)
        out.append(ok)
    return np.stack(out, 1)


def coverage(W, D, camera, height, pitch_range, boxes=(), n=6000):
    xy = rng.uniform([0.3, 0.3], [W - 0.3, D - 0.3], (n, 2))
    for b in boxes:  # no cameras inside furniture
        xy = xy[~((xy[:, 0] > b[0]) & (xy[:, 0] < b[3]) & (xy[:, 1] > b[1]) & (xy[:, 1] < b[4]))]
    c = np.column_stack([xy, np.full(len(xy), height)])
    yaw = rng.uniform(-np.pi, np.pi, len(xy))
    pitch = np.radians(rng.uniform(*pitch_range, len(xy)))
    res = {}
    for name, ms in layouts(W, D).items():
        k = visible(c, yaw, pitch, ms, camera, boxes).sum(1)
        res[name] = {"ge1": round(100 * (k >= 1).mean(), 1), "ge2": round(100 * (k >= 2).mean(), 1)}
        if camera is DUCK:  # duck turns on the spot (12 headings): can it find a marker from here at all?
            spin = np.zeros(len(c), bool)
            for a in np.linspace(-np.pi, np.pi, 12, endpoint=False):
                spin |= visible(c, np.full(len(c), a), pitch, ms, camera, boxes).any(1)
            res[name]["spin"] = round(100 * spin.mean(), 1)
    return res


def longest_blind(path, vis_any):
    seg = np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)]
    best = run = 0.0
    for s, v in zip(seg, vis_any):
        run = 0.0 if v else run + s
        best = max(best, run)
    return round(best, 2)


def walk(points, step=0.05):
    pts = [np.linspace(a, b, max(2, int(np.linalg.norm(np.subtract(b, a)) / step)), endpoint=False)
           for a, b in zip(points[:-1], points[1:])]
    p = np.vstack(pts)
    d = np.diff(np.vstack([p, points[-1]]), axis=0)
    return p, np.arctan2(d[:, 1], d[:, 0])


def duck_walks(boxes):
    res = {}
    # Loop 0.6 m inside a 4x4 room, duck looks where it walks.
    loop, yaw = walk([(0.6, 0.6), (3.4, 0.6), (3.4, 3.4), (0.6, 3.4), (0.6, 0.6)])
    c = np.column_stack([loop, np.full(len(loop), 0.2)])
    wl = list(walls(4, 4).values())
    res["loop"] = {N: longest_blind(loop, visible(c, yaw, np.zeros(len(c)), wl[:N], DUCK, boxes).any(1)) for N in (1, 2, 3, 4)}
    # Two 4x4 rooms side by side, shared wall x=4 with a door at y 1.6-2.4. Walk A(1,2) -> B(7,2) and back through the door.
    shared = [(4 - 0.04, 0, 0, 4 + 0.04, 1.6, 1.6), (4 - 0.04, 2.4, 0, 4 + 0.04, 4, 1.6)]
    path, yaw = walk([(1, 2), (7, 2), (7, 2.2), (1, 2.2)])
    c = np.column_stack([path, np.full(len(path), 0.2)])
    roomA = [((4, 3.0), (-1, 0)), ((0, 2), (1, 0)), ((2, 0), (0, 1)), ((2, 4), (0, -1))]  # door wall first, then far wall
    roomB = [((4, 3.0), (1, 0)), ((8, 2), (-1, 0)), ((6, 0), (0, 1)), ((6, 4), (0, -1))]
    res["doorway"] = {N: longest_blind(path, visible(c, yaw, np.zeros(len(c)), roomA[:N] + roomB[:N], DUCK, shared).any(1))
                      for N in (1, 2, 3, 4)}
    return res


def flip_test(dist, view_deg, noise_px=0.5, trials=2000):
    """Duck camera at `dist` from a wall, `view_deg` off the marker normal. How often is the solved
    camera position more than 30 cm off? One 15 cm marker (IPPE square) vs two markers 1 m apart (8 corners)."""
    import cv2 as cv
    f = DUCK[0]
    K = np.array([[f, 0, 320], [0, f, 180], [0, 0, 1]])
    s2 = SIZE / 2
    sq = np.array([[-s2, s2, 0], [s2, s2, 0], [s2, -s2, 0], [-s2, -s2, 0]], np.float32)
    two = np.vstack([sq + [-0.5, 0, 0], sq + [0.5, 0, 0]]).astype(np.float32)
    a = np.radians(view_deg)  # camera sits on a circle around the wall centre, looking at it
    cam_w = np.array([dist * np.sin(a), 0, dist * np.cos(a)])  # marker frame: wall = z 0 plane, normal +z
    zc = -cam_w / dist
    xc = np.cross([0, 1, 0], zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    rvec, _ = cv.Rodrigues(R)
    tvec = -R @ cam_w
    out = {}
    for name, obj in (("one marker", sq), ("pair 1 m apart", two)):
        img, _ = cv.projectPoints(obj, rvec, tvec, K, None)
        bad = 0
        for _ in range(trials):
            noisy = (img.reshape(-1, 2) + rng.normal(0, noise_px, (len(obj), 2))).astype(np.float32)
            if len(obj) == 4:
                _, rs, ts, _ = cv.solvePnPGeneric(obj, noisy, K, None, flags=cv.SOLVEPNP_IPPE_SQUARE)
                r, t = rs[0], ts[0]  # best-reprojection solution, as a naive tracker would take
            else:
                _, r, t = cv.solvePnP(obj, noisy, K, None, flags=cv.SOLVEPNP_SQPNP)
            Rm, _ = cv.Rodrigues(r)
            bad += np.linalg.norm(-Rm.T @ t.ravel() - cam_w) > 0.30
        out[name] = round(100 * bad / trials, 1)
    return out


sofa = [(1.0, 1.4, 0, 3.0, 2.3, 0.45)]  # free-standing sofa in the middle of a 4x4 room
results = {
    "phone_chest_4x4": coverage(4, 4, PHONE, 1.3, (-30, -5)),
    "phone_knee_4x4": coverage(4, 4, PHONE, 0.5, (-15, 5)),
    "phone_chest_3x5": coverage(3, 5, PHONE, 1.3, (-30, -5)),
    "duck_4x4": coverage(4, 4, DUCK, 0.2, (0, 0)),
    "duck_4x4_sofa": coverage(4, 4, DUCK, 0.2, (0, 0), sofa),
    "duck_3x5": coverage(3, 5, DUCK, 0.2, (0, 0)),
    "duck_blind_m": duck_walks(sofa),
    "pos_err_over_30cm_pct": {f"{d} m, {v} deg": flip_test(d, v) for d in (1.0, 2.0, 3.0) for v in (0, 20, 45)},
    "phone_focal_px": round(PHONE[0]),
    "duck_max_range_m": round(DUCK[0] * SIZE / MIN_PX, 2),
    "phone_max_range_m": round(PHONE[0] * SIZE / MIN_PX, 2),
}
print(json.dumps(results, indent=1))
