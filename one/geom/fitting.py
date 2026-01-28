import numpy as np
import one.geom.geometry as ogg


def convex_hull(geom):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(geom.vs)
    vs = geom.vs
    fs = hull.simplices.astype(np.uint32)
    # ensure outward normals
    # TODO: extract function?
    center = vs.mean(axis=0)
    v0 = vs[fs[:, 0]]
    v1 = vs[fs[:, 1]]
    v2 = vs[fs[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    p = (v0 + v1 + v2) / 3.0
    mask = np.einsum('ij,ij->i', p - center, n) < 0
    fs[mask] = fs[mask][:, [0, 2, 1]]
    return ogg.gen_geom_from_raw(vs, fs)


def fit_plane_from_pts(pts):
    c = pts.mean(axis=0)
    cov = (pts - c).T @ (pts - c) / max(len(pts), 1)
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]  # smallest eigenvalue
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -np.dot(n, c)
    return n.astype(np.float32), float(d)
