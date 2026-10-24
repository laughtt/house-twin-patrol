"""Every frame of a scan into an OpenCV 5 TSDF and a 2.5 cm voxel grid, then the house frame: floor at z 0, walls on
the x/y axes (Manhattan). Later stages work on `Scene.V` (voxel centres), `Scene.Nv` (normals) and `Scene.traj`."""
from types import SimpleNamespace

import cv2 as cv
import numpy as np

from twin.scan_io import MAX_D, kmat

VOX = 0.025


def vkey(p, vox=VOX):
    k = np.floor(p / vox).astype(np.int64) + (1 << 17)
    return (k[:, 0] << 36) | (k[:, 1] << 18) | k[:, 2]


def means(inv, n, *arrays):
    cnt = np.bincount(inv, minlength=n)
    return cnt, [np.stack([np.bincount(inv, a[:, j], n) for j in range(a.shape[1])], 1) / np.maximum(cnt, 1)[:, None]
                 for a in arrays]


def lowest_layer(z, lo=-3.0, hi=1.0):
    hist, edges = np.histogram(z, bins=np.arange(lo, hi, 0.02))
    hist = np.convolve(hist, np.ones(3), "same")
    z0 = edges[np.nonzero(hist >= 0.15 * hist.max())[0][0]] + 0.01
    return float(np.median(z[np.abs(z - z0) < 0.04]))


def manhattan(n):
    # wall normals' direction mod 90°: histogram mode, then a 4θ circular mean around it
    phi = np.arctan2(n[:, 1], n[:, 0])
    hist = np.bincount((np.degrees(phi) % 90).astype(int) % 90, minlength=90)
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid")
    mode = np.radians(np.argmax(hist))
    near = np.abs((phi - mode + np.pi / 4) % (np.pi / 2) - np.pi / 4) < np.radians(6)
    return float(np.angle(np.exp(4j * phi[near]).sum()) / 4) % (np.pi / 2)


def ground_map(V, Nv, tile=0.5):
    """Floor height under every 50 cm tile (homes have steps and sunken rooms): the low end of what faces up there.
    Tiles with no floor in view (under a bed or a sofa) take the mean of the known tiles around them."""
    up = V[(Nv[:, 2] > 0.9) & (V[:, 2] > -0.1) & (V[:, 2] < 0.35)]
    lo = V[:, :2].min(0)
    shape = tuple(np.ceil((V[:, :2].max(0) - lo) / tile).astype(int)[::-1] + 1)
    ij = np.floor((up[:, :2] - lo) / tile).astype(int)
    key = ij[:, 1] * shape[1] + ij[:, 0]
    order = np.lexsort((up[:, 2], key))
    k, first, cnt = np.unique(key[order], return_index=True, return_counts=True)
    g = np.full(shape, np.nan, np.float32)
    g.flat[k[cnt >= 30]] = up[order, 2][(first + cnt // 5)[cnt >= 30]]  # 20th percentile of each tile
    known = ~np.isnan(g)
    if not known.any():
        g[:], known[:] = 0, True
    while not known.all():
        v = cv.blur(np.where(known, g, 0), (3, 3), borderType=cv.BORDER_CONSTANT)
        n = cv.blur(known.astype(np.float32), (3, 3), borderType=cv.BORDER_CONSTANT)
        grow = ~known & (n > 0)
        g[grow], known = v[grow] / n[grow], known | grow
    return lo, tile, cv.medianBlur(g, 3)


def ground_at(gm, xy):
    lo, tile, g = gm
    ij = np.clip(np.floor((np.asarray(xy, float) - lo) / tile).astype(int), 0, [g.shape[1] - 1, g.shape[0] - 1])
    return g[ij[..., 1], ij[..., 0]]


def fuse(ex, frames):
    w, h = frames[0]["depth_wh"]
    Kd = kmat(frames[0]["K_depth"])
    vs = cv.VolumeSettings(cv.VolumeType_HashTSDF)
    vs.setIntegrateWidth(w), vs.setIntegrateHeight(h), vs.setRaycastWidth(w), vs.setRaycastHeight(h)
    vs.setCameraIntegrateIntrinsics(Kd), vs.setCameraRaycastIntrinsics(Kd)
    vs.setVoxelSize(VOX), vs.setTsdfTruncateDistance(4 * VOX), vs.setDepthFactor(1.0), vs.setMaxDepth(MAX_D)
    vol = cv.Volume(cv.VolumeType_HashTSDF, vs)
    normals = cv.RgbdNormals_create(h, w, cv.CV_32F, Kd, 5, method=cv.RGBD_NORMALS_RGBD_NORMALS_METHOD_FALS)
    names = sorted({f["room"] for f in frames}, key=[f["room"] for f in frames].index)
    pts, nrm, col, lab, poses = [], [], [], [], []
    for f in frames:
        d, P = ex.load(f)
        vol.integrate(d, P.astype(np.float32))
        P3 = cv.depthTo3d(d, Kd)
        N = normals.apply(P3)[::2, ::2, :3].reshape(-1, 3)
        q = P3[::2, ::2, :3].reshape(-1, 3)
        ok = (d[::2, ::2].ravel() > 0) & np.isfinite(N).all(1)
        q, N = q[ok], N[ok]
        N *= -np.sign((N * q).sum(1))[:, None]  # face the camera
        pts.append(q @ P[:3, :3].T + P[:3, 3])
        nrm.append(N @ P[:3, :3].T)
        col.append(cv.resize(ex.rgb(f), (w, h), interpolation=cv.INTER_AREA)[::2, ::2].reshape(-1, 3)[ok])
        lab.append(np.full(len(q), names.index(f["room"]), np.int8))
        poses.append(P)
    # voxel grid: mean position / normal / colour and the majority room label of the frames that saw it
    uk, inv = np.unique(vkey(np.concatenate(pts)), return_inverse=True)
    cnt, (V, Nv, C) = means(inv, len(uk), np.concatenate(pts), np.concatenate(nrm), np.concatenate(col).astype(float))
    votes = np.bincount(inv * len(names) + np.concatenate(lab), minlength=len(uk) * len(names)).reshape(-1, len(names))
    keep = cnt >= 2
    V, Nv, C, L = V[keep], Nv[keep], C[keep], votes[keep].argmax(1)
    Nv /= np.linalg.norm(Nv, axis=1, keepdims=True) + 1e-9
    del pts, nrm, col, lab
    z0 = lowest_layer(V[Nv[:, 2] > 0.9, 2])
    V[:, 2] -= z0
    wall = (np.abs(Nv[:, 2]) < 0.3) & (V[:, 2] > 0.3) & (V[:, 2] < 2.0)
    theta = manhattan(Nv[wall])
    R2 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    V[:, :2], Nv[:, :2] = V[:, :2] @ R2.T, Nv[:, :2] @ R2.T
    traj = np.array([P[:3, 3] for P in poses])
    traj[:, :2] = traj[:, :2] @ R2.T
    traj[:, 2] -= z0
    gm = ground_map(V, Nv)  # from here on every height is above the floor right under it
    V[:, 2] -= ground_at(gm, V[:, :2])
    traj[:, 2] -= ground_at(gm, traj[:, :2])
    return SimpleNamespace(V=V, Nv=Nv, C=C, L=L, wall=wall, names=names, tlab=[f["room"] for f in frames], poses=poses,
                           traj=traj, z0=z0, theta=theta, R2=R2, vol=vol, Kd=Kd, wh=(w, h), gm=gm, ground=lambda xy: ground_at(gm, xy))


def to_house(sc, p, o):
    """ARKit-derived world points (n, 3) → the house frame (floor z 0, axes aligned, origin o)."""
    p = np.array(p, float)
    p[:, :2] = p[:, :2] @ sc.R2.T
    p[:, 2] -= sc.z0 + sc.ground(p[:, :2])
    p[:, :2] -= o
    return p


def cloud(path, sc, o):
    """The fused TSDF surface, raycast from every 8th pose, 3 cm → PLY coloured from the voxel grid (5 cm lookup)."""
    tp, tn = [], []
    for P in sc.poses[::8]:
        q, n = (a.reshape(-1, 4)[:, :3].astype(float) for a in sc.vol.raycast(P.astype(np.float32)))
        ok = np.isfinite(q).all(1) & (q[:, 2] > 0)
        tp.append(q[ok] @ P[:3, :3].T + P[:3, 3])
        tn.append(n[ok] @ P[:3, :3].T)
    tp, tn = np.concatenate(tp), np.concatenate(tn)
    tk, tinv = np.unique(vkey(tp, 0.03), return_inverse=True)
    _, (tp, tn) = means(tinv, len(tk), tp, tn)
    tp = to_house(sc, tp, o)
    tn[:, :2] = tn[:, :2] @ sc.R2.T
    V = sc.V.copy()
    V[:, :2] -= o
    rgb, miss = np.full((len(tp), 3), 0.6), np.ones(len(tp), bool)
    for vox in (0.05, 0.1, 0.2):  # colour of the voxel there, else of the 10 cm / 20 cm block around it
        ck, cinv = np.unique(vkey(V, vox), return_inverse=True)
        _, (cc,) = means(cinv, len(ck), sc.C)
        q = vkey(tp[miss], vox)
        j = np.clip(np.searchsorted(ck, q), 0, len(ck) - 1)
        hit = ck[j] == q
        idx = np.nonzero(miss)[0][hit]
        rgb[idx], miss[idx] = cc[j[hit]][:, ::-1] / 255, False
    cv.savePointCloud(str(path), *(a.astype(np.float32).reshape(-1, 1, 3) for a in (tp, tn, rgb)))
    # the same points for a web viewer: header (lo xyz float32, count uint32), xyz in cm above lo (uint16), rgb (uint8)
    lo = tp.min(0)
    with open(path.with_suffix(".bin"), "wb") as f:
        f.write(np.r_[lo].astype(np.float32).tobytes() + np.uint32(len(tp)).tobytes())
        f.write(np.round((tp - lo) * 100).clip(0, 65535).astype(np.uint16).tobytes() + (rgb * 255).astype(np.uint8).tobytes())
    return tp
