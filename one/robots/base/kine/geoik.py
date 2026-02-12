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