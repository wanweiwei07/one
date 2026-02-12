from dataclasses import dataclass

import numpy as np

import one.utils.math as oum


@dataclass
class KineInfo:
    parallel_pairs: list
    intersect_pairs: list
    spherical_wrist: bool
    three_parallel_groups: list
    wrist_center: np.ndarray | None = None


def is_parallel(h1, h2, eps=1e-6):
    return np.linalg.norm(np.cross(h1, h2)) < eps


def line_line_distance(p1, d1, p2, d2):
    """
    distance between two 3D lines:
    L1: p1 + t d1
    L2: p2 + s d2
    """
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    cross = np.cross(d1, d2)
    denom = np.linalg.norm(cross)
    if denom < 1e-9:
        return np.linalg.norm(np.cross(p2 - p1, d1))
    return abs(np.dot(p2 - p1, cross)) / denom


def line_point_distance(p, l0, d):
    d = d / np.linalg.norm(d)
    return np.linalg.norm(np.cross(p - l0, d))


def line_line_midpoint(p1, d1, p2, d2, eps=1e-9):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    a11 = np.dot(d1, d1)
    a12 = -np.dot(d1, d2)
    a21 = np.dot(d1, d2)
    a22 = -np.dot(d2, d2)
    det = a11 * a22 - a12 * a21
    if abs(det) < eps:
        return None
    rhs1 = np.dot(p2 - p1, d1)
    rhs2 = np.dot(p2 - p1, d2)
    t, s = np.linalg.solve(np.array([[a11, a12], [a21, a22]], dtype=np.float64),
                           np.array([rhs1, rhs2], dtype=np.float64))
    c1 = p1 + t * d1
    c2 = p2 + s * d2
    return (c1 + c2) * 0.5


def estimate_wrist_center(o1, h1, o2, h2, o3, h3, eps_intersect=1e-5):
    c12 = line_line_midpoint(o1, h1, o2, h2)
    c23 = line_line_midpoint(o2, h2, o3, h3)
    c13 = line_line_midpoint(o1, h1, o3, h3)
    if c12 is None or c23 is None or c13 is None:
        return None
    center = (c12 + c23 + c13) / 3.0
    if (line_point_distance(center, o1, h1) < eps_intersect
            and line_point_distance(center, o2, h2) < eps_intersect
            and line_point_distance(center, o3, h3) < eps_intersect):
        return center
    return None


def is_intersecting(p1, d1, p2, d2, eps=1e-5):
    return line_line_distance(p1, d1, p2, d2) < eps


def analyze_decomposition(chain, eps_parallel=1e-6, eps_intersect=1e-5):
    n = len(chain.jnts)
    parallel_pairs = []
    intersect_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = chain.axes[i], chain.axes[j]
            oi, oj = chain.origins[i], chain.origins[j]
            if is_parallel(hi, hj, eps_parallel):
                parallel_pairs.append((i, j))
            elif is_intersecting(oi, hi, oj, hj, eps_intersect):
                intersect_pairs.append((i, j))
    spherical_wrist = False
    wrist_center = None
    if n >= 6:
        i, j, k = n - 3, n - 2, n - 1
        if (is_intersecting(chain.origins[i], chain.axes[i], chain.origins[j], chain.axes[j], eps_intersect)
                and is_intersecting(chain.origins[j], chain.axes[j], chain.origins[k], chain.axes[k], eps_intersect)
                and is_intersecting(chain.origins[i], chain.axes[i], chain.origins[k], chain.axes[k], eps_intersect)):
            wrist_center = estimate_wrist_center(chain.origins[i], chain.axes[i],
                                                 chain.origins[j], chain.axes[j],
                                                 chain.origins[k], chain.axes[k],
                                                 eps_intersect=eps_intersect)
            spherical_wrist = wrist_center is not None
    three_parallel_groups = []
    for i in range(n - 2):
        if (is_parallel(chain.axes[i], chain.axes[i + 1], eps_parallel)
                and is_parallel(chain.axes[i + 1], chain.axes[i + 2], eps_parallel)):
            three_parallel_groups.append((i, i + 1, i + 2))
    return KineInfo(parallel_pairs=parallel_pairs,
                    intersect_pairs=intersect_pairs,
                    spherical_wrist=spherical_wrist,
                    three_parallel_groups=three_parallel_groups,
                    wrist_center=wrist_center)


def _as_unit(v):
    return oum.unit_vec(v, return_length=False)


def rotate_about_axis(axis, theta, point, axis_point=None):
    axis = _as_unit(axis)
    point = np.asarray(point, dtype=np.float32)
    if axis_point is None:
        axis_point = np.zeros(3, dtype=np.float32)
    else:
        axis_point = np.asarray(axis_point, dtype=np.float32)
    rel = point - axis_point
    rel_rot = oum.rotmat_from_axangle(axis, theta) @ rel
    return (axis_point + rel_rot).astype(np.float32)


def sp1(axis, p, q, tol=1e-8):
    axis = _as_unit(axis)
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    pp = p - np.dot(axis, p) * axis
    qp = q - np.dot(axis, q) * axis
    npp = np.linalg.norm(pp)
    nqp = np.linalg.norm(qp)
    if npp < tol or nqp < tol:
        return []
    if abs(np.dot(axis, p) - np.dot(axis, q)) > 1e-5:
        return []
    pp_u = pp / npp
    qp_u = qp / nqp
    c = np.clip(np.dot(pp_u, qp_u), -1.0, 1.0)
    s = np.dot(axis, np.cross(pp_u, qp_u))
    return [float(oum.wrap_to_pi(np.arctan2(s, c)))]


def sp3(axis, p, q, d, tol=1e-8):
    axis = _as_unit(axis)
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    d = float(d)
    hp = np.dot(axis, p)
    hq = np.dot(axis, q)
    dp_sq = d * d - (hp - hq) * (hp - hq)
    if dp_sq < -1e-8:
        return []
    dp_sq = max(0.0, dp_sq)
    up = p - hp * axis
    uq = q - hq * axis
    A = 2.0 * np.dot(up, uq)
    B = 2.0 * np.dot(axis, np.cross(up, uq))
    C = np.dot(up, up) + np.dot(uq, uq) - dp_sq
    R = np.hypot(A, B)
    if R < tol:
        if abs(C) < 1e-7:
            return [0.0]
        return []
    if abs(C) > R + 1e-7:
        return []
    phi = np.arctan2(B, A)
    delta = np.arccos(np.clip(C / R, -1.0, 1.0))
    if delta < 1e-8:
        return [float(oum.wrap_to_pi(phi))]
    return [float(oum.wrap_to_pi(phi + delta)),
            float(oum.wrap_to_pi(phi - delta))]