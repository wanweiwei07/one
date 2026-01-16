import numpy as np


def build_triangles(vertices, faces):
    """(V,3) + (F,3) -> (F,3,3)"""
    return vertices[faces]


def compute_mesh_aabb(tris):
    mins = tris.min(axis=(0, 1))
    maxs = tris.max(axis=(0, 1))
    return mins, maxs


def aabb_intersect(min_a, max_a, min_b, max_b):
    return np.all(min_a <= max_b) and np.all(max_a >= min_b)


def compute_triangle_planes(tris):
    """
    tris: (N,3,3)
    return:
        normals: (N,3)
        offsets: (N,)
    """
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(edge1, edge2)
    offsets = -np.einsum("ij,ij->i", normals, tris[:, 0])
    return normals, offsets


def tripair_planeprojection_filter(tris_a, tris_b, eps=1e-4):
    """
    Find first intersecting triangle pair.
    tris_a, tris_b: (N,3,3)
    return: (hit_found, (ia, ib)) or (False, None)
    """
    # plane tests
    normals_a, offsets_a = compute_triangle_planes(tris_a)
    normals_b, offsets_b = compute_triangle_planes(tris_b)
    dist_to_plane_a = (
            np.einsum("nij,nj->ni", tris_b, normals_a) +
            offsets_a[:, None])
    dist_to_plane_b = (
            np.einsum("nij,nj->ni", tris_a, normals_b) +
            offsets_b[:, None])
    separated_by_a = (
            (dist_to_plane_a > eps).all(axis=1) |
            (dist_to_plane_a < -eps).all(axis=1))
    separated_by_b = (
            (dist_to_plane_b > eps).all(axis=1) |
            (dist_to_plane_b < -eps).all(axis=1))
    candidate_mask = ~(separated_by_a | separated_by_b)
    if not np.any(candidate_mask):
        return False, None
    # filter candidates
    idx = np.where(candidate_mask)[0]
    tris_a = tris_a[idx]
    tris_b = tris_b[idx]
    normals_a = normals_a[idx]
    normals_b = normals_b[idx]
    # intersection direction tests
    intersection_dirs = np.cross(normals_a, normals_b)
    dir_norms = np.linalg.norm(intersection_dirs, axis=1)
    non_coplanar = dir_norms > eps
    proj_a = np.einsum("nij,nj->ni", tris_a, intersection_dirs)
    proj_b = np.einsum("nij,nj->ni", tris_b, intersection_dirs)
    min_a = proj_a.min(axis=1)
    max_a = proj_a.max(axis=1)
    min_b = proj_b.min(axis=1)
    max_b = proj_b.max(axis=1)
    interval_overlap = (min_a <= max_b) & (max_a >= min_b)
    hit_local = interval_overlap & non_coplanar
    if not np.any(hit_local):
        return False, None
    tris_a_hit = tris_a[hit_local]
    tris_b_hit = tris_b[hit_local]
    return True, (tris_a_hit, tris_b_hit)


def tripair_fine_filter(tris_a, tris_b, eps=1e-9):
    # tris_a, tris_b: (N,3,3)
    nA = np.cross(tris_a[:, 1] - tris_a[:, 0], tris_a[:, 2] - tris_a[:, 0])
    nB = np.cross(tris_b[:, 1] - tris_b[:, 0], tris_b[:, 2] - tris_b[:, 0])
    nA /= (np.linalg.norm(nA, axis=1, keepdims=True) + eps)
    nB /= (np.linalg.norm(nB, axis=1, keepdims=True) + eps)
    dA = -np.einsum("ij,ij->i", nA, tris_a[:, 0])
    dB = -np.einsum("ij,ij->i", nB, tris_b[:, 0])
    dir_vec = np.cross(nA, nB)
    dir_norm = np.linalg.norm(dir_vec, axis=1)
    valid = dir_norm >= 1e-4
    # p0 shape (N,3)
    p0 = np.cross((dB[:, None] * nA - dA[:, None] * nB),
                  dir_vec) / (dir_norm[:, None] ** 2 + eps)
    dir_vec = dir_vec / (dir_norm[:, None] + eps)
    proj_a = np.einsum("nij,nj->ni", tris_a, dir_vec)
    proj_b = np.einsum("nij,nj->ni", tris_b, dir_vec)
    t0 = np.maximum(proj_a.min(axis=1), proj_b.min(axis=1))
    t1 = np.minimum(proj_a.max(axis=1), proj_b.max(axis=1))
    valid &= (t0 < t1)
    points = p0 + dir_vec * ((t0 + t1) * 0.5)[:, None]
    # TODO: check if points are within both triangles
    inside_a = _point_in_tri_batch(points, tris_a)
    inside_b = _point_in_tri_batch(points, tris_b)
    valid &= inside_a & inside_b
    return points, valid


def is_sobj_collided(sobj_a, sobj_b, eps=1e-9):
    merged_a = _merge_collisions(sobj_a)
    merged_b = _merge_collisions(sobj_b)
    if merged_a is None or merged_b is None:
        return None
    verts_a, faces_a = merged_a
    verts_b, faces_b = merged_b
    tris_a = verts_a[faces_a]
    tris_b = verts_b[faces_b]
    # mesh AABB early-out
    min_a = tris_a.min(axis=(0, 1))
    max_a = tris_a.max(axis=(0, 1))
    min_b = tris_b.min(axis=(0, 1))
    max_b = tris_b.max(axis=(0, 1))
    if not aabb_intersect(min_a, max_a, min_b, max_b):
        return None
    # tri AABB overlap pairs
    tri_min_a = tris_a.min(axis=1)
    tri_max_a = tris_a.max(axis=1)
    tri_min_b = tris_b.min(axis=1)
    tri_max_b = tris_b.max(axis=1)
    overlap_mask = (
            (tri_min_a[:, None, :] <= tri_max_b[None, :, :]) &
            (tri_max_a[:, None, :] >= tri_min_b[None, :, :])
    ).all(axis=-1)
    pair_indices = np.argwhere(overlap_mask)
    if len(pair_indices) == 0:
        return None
    tris_a_pairs = tris_a[pair_indices[:, 0]]
    tris_b_pairs = tris_b[pair_indices[:, 1]]
    hit, tris_pairs = tripair_planeprojection_filter(tris_a_pairs, tris_b_pairs, eps=eps)
    if not hit:
        return None
    points, valid = tripair_fine_filter(tris_pairs[0], tris_pairs[1], eps=eps)
    if not np.any(valid):
        return None
    return points[valid]


def _merge_collisions(sobj):
    verts_all = []
    faces_all = []
    offset = 0
    for col in sobj.collisions:
        geom = col.geometry
        if geom is None or geom.faces is None:
            continue
        verts = geom.verts
        faces = geom.faces
        # world transform: sobj.node.wd_tf @ col._tfmat
        world_tf = sobj.node.wd_tf @ col._tf
        R = world_tf[:3, :3]
        t = world_tf[:3, 3]
        verts_w = (R @ verts.T).T + t
        verts_all.append(verts_w)
        faces_all.append(faces + offset)
        offset += verts.shape[0]
    if not verts_all:
        return None
    verts_all = np.vstack(verts_all)
    faces_all = np.vstack(faces_all)
    return verts_all, faces_all


def _point_in_tri_batch(pts, tris, eps=1e-9):
    v0 = tris[:, 2] - tris[:, 0]
    v1 = tris[:, 1] - tris[:, 0]
    v2 = pts - tris[:, 0]
    dot00 = np.einsum("ij,ij->i", v0, v0)
    dot01 = np.einsum("ij,ij->i", v0, v1)
    dot02 = np.einsum("ij,ij->i", v0, v2)
    dot11 = np.einsum("ij,ij->i", v1, v1)
    dot12 = np.einsum("ij,ij->i", v1, v2)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + eps)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)
