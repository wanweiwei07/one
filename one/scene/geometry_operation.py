import numpy as np
import one.utils.math as oum


def revolve(profile, segments=36):
    """
    Revolve a 2D profile (r,z) around Z-axis to produce a mesh
    author: weiwei
    date: 20251201
    """
    profile = np.asarray(profile, dtype=np.float32)
    if np.any(profile[1:-1, 0] <= oum.eps):
        raise ValueError("Only the first and last radius may be zero! Others must be positive.")
    has_bottom = (profile[0, 0] <= oum.eps)
    has_top = (profile[-1, 0] <= oum.eps)
    profile_core = profile[1:] if has_bottom else profile
    profile_core = profile_core[:-1] if has_top else profile_core
    r_core = profile_core[:, 0]
    z_core = profile_core[:, 1]
    n_core = len(r_core)
    # angles
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # verts
    r_mat = np.tile(r_core, (segments, 1))
    z_mat = np.tile(z_core, (segments, 1))
    x = r_mat * cos_t[:, None]
    y = r_mat * sin_t[:, None]
    verts = np.stack([x, y, z_mat], axis=2).reshape(-1, 3)
    # side wall faces
    idx = np.arange(segments * n_core, dtype=np.uint32).reshape(segments, n_core)
    idx_next = np.roll(idx, -1, axis=0)
    # tri1 = (i,j), (i+1,j), (i,j+1)
    tri1 = np.stack([idx[:, :-1], idx_next[:, :-1], idx[:, 1:]], axis=2).reshape(-1, 3)
    # tri2 = (i,j+1), (i+1,j), (i+1,j+1)
    tri2 = np.stack([idx[:, 1:], idx_next[:, :-1], idx_next[:, 1:]], axis=2).reshape(-1, 3)
    faces_list = [tri1, tri2]
    extra_verts = []
    # bottom cap
    ring_bottom_idx = idx[:, 0]  # shape (seg,)
    verts_bottom = verts[ring_bottom_idx]
    if has_bottom:
        center_bottom = np.array([0.0, 0.0, profile[0, 1]], dtype=np.float32)
    else:
        center_bottom = verts_bottom.mean(axis=0)
    center_bottom_idx = len(verts)  # new idx for center vertex
    extra_verts.append(center_bottom)
    tri_bottom = np.stack([np.full(segments, center_bottom_idx, dtype=np.uint32), np.roll(ring_bottom_idx, -1),
                           ring_bottom_idx], axis=1)
    faces_list.append(tri_bottom)
    # top cap
    ring_top_idx = idx[:, -1]  # shape (seg,)
    verts_top = verts[ring_top_idx]
    if has_top:
        center_top = np.array([0.0, 0.0, profile[-1, 1]], dtype=np.float32)
    else:
        center_top = verts_top.mean(axis=0)
    center_top_idx = len(verts) + len(extra_verts)  # new idx for center vertex
    extra_verts.append(center_top)
    tri_top = np.stack([np.full(segments, center_top_idx, dtype=np.uint32), ring_top_idx,
                        np.roll(ring_top_idx, -1)], axis=1)
    faces_list.append(tri_top)
    verts = np.vstack([verts, np.vstack(extra_verts)])
    faces = np.concatenate(faces_list, axis=0)
    return (verts, faces)


def subdivide_once(verts, faces):
    edges = np.sort(np.stack([faces[:, [0, 1]],
                              faces[:, [1, 2]],
                              faces[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
    # unique edges and their midpoints
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    mid = verts[unique].mean(axis=1)
    mid_idx = inverse.reshape(-1, 3) + len(verts)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m0, m1, m2 = mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]
    new_faces = np.vstack([np.column_stack([v0, m0, m2]),
                           np.column_stack([v1, m1, m0]),
                           np.column_stack([v2, m2, m1]),
                           np.column_stack([m0, m1, m2])])
    new_verts = np.vstack([verts, mid])
    new_verts /= np.linalg.norm(new_verts, axis=1, keepdims=True)
    return new_verts, new_faces


def icosahedron():
    """Generate vertices and faces of a unit icosahedron."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.uint32)
    # normalize (unit sphere)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, faces


def sample_surface(vs, fs, n_samples):
    v0 = vs[fs[:, 0]]
    v1 = vs[fs[:, 1]]
    v2 = vs[fs[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    prob = areas / np.sum(areas)
    fids = np.random.choice(len(fs), size=n_samples, p=prob)
    r1 = np.sqrt(np.random.rand(n_samples))
    r2 = np.random.rand(n_samples)
    a = 1 - r1
    b = r1 * (1 - r2)
    c = r1 * r2
    pts = a[:, None] * v0[fids] + b[:, None] * v1[fids] + c[:, None] * v2[fids]
    normals = np.cross(v1[fids] - v0[fids],
                       v2[fids] - v0[fids])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return pts, normals, fids


def segment_surface(geometry, normal_tol_deg=15.0):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    # build adjacency graph
    fs = geometry.fs
    edges = np.stack([
        np.sort(fs[:, [0, 1]], axis=1),
        np.sort(fs[:, [1, 2]], axis=1),
        np.sort(fs[:, [2, 0]], axis=1)
    ], axis=1).reshape(-1, 2)  # (F*3, 2)
    fids = np.repeat(np.arange(len(fs)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[order]
    face_sorted = fids[order]
    same = np.all(edges_sorted[1:] == edges_sorted[:-1], axis=1)
    idx = np.where(same)[0]
    i = face_sorted[idx]
    j = face_sorted[idx + 1]
    # filter by normal angle
    fns = geometry.fns
    cos_th = np.cos(np.deg2rad(normal_tol_deg))
    cos_ij = np.einsum('ij,ij->i', fns[i], fns[j])
    keep = cos_ij >= cos_th
    i = i[keep]
    j = j[keep]
    # adjacency by plane distance
    data = np.ones(len(i) * 2, dtype=np.uint8)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    A = coo_matrix((data, (rows, cols)),
                   shape=(len(fs), len(fs)))
    n_comp, labels = connected_components(A, directed=False)
    parts = []
    for c in range(n_comp):
        fids_c = np.where(labels == c)[0]
        if fids_c.size == 0:
            continue
        parts.append(fids_c)
    return tuple(parts)


def convex_hull(geom):
    from scipy.spatial import ConvexHull
    import one.scene.geometry as osg
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
    return osg.Geometry(vs, fs)


def ray_shoot_flat(orig, dir, verts, faces,
                   face_normals, eps=oum.eps):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # AABB mask
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)
    inv_dir = 1.0 / np.where(dir == 0, oum.eps, dir)
    t1 = (tri_min - orig) * inv_dir
    t2 = (tri_max - orig) * inv_dir
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    mask = tmax >= np.maximum(tmin, 0.0)
    if not np.any(mask):
        return None
    ids = np.where(mask)[0]
    v0 = v0[ids]
    v1 = v1[ids]
    v2 = v2[ids]
    # Moller-Trumbore
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(dir, e2)
    a = np.sum(e1 * h, axis=1)
    m = np.abs(a) > eps
    f = np.zeros_like(a)
    f[m] = 1.0 / a[m]
    s = orig - v0
    u = f * np.sum(s * h, axis=1)
    m &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(s, e1)
    v = f * np.sum(dir * q, axis=1)
    m &= (v >= 0.0) & (u + v <= 1.0)
    # t is the ray parameter: P = orig + t * dir
    t = f * np.sum(e2 * q, axis=1)
    m &= (t > eps)
    hit_ids = ids[m]
    if len(hit_ids) == 0:
        return None
    hit_t = t[m]
    hit_pos = orig + hit_t[:, None] * dir
    hit_n = face_normals[hit_ids]
    order = np.argsort(hit_t)
    return (hit_pos[order], hit_n[order],
            hit_t[order], hit_ids[order])


def ray_shoot(orig, dir, geometry):
    return ray_shoot_flat(
        orig, dir, geometry.vs,
        geometry.fs, geometry.fns)
