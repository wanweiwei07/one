import one.utils.math as rm

def revolve(profile, segments=36):
    """
    Revolve a 2D profile (r,z) around Z-axis to produce a mesh
    author: weiwei
    date: 20251201
    """
    profile = rm.asarray(profile, dtype=rm.float32)
    if rm.any(profile[1:-1, 0] <= rm.eps):
        raise ValueError("Only the first and last radius may be zero! Others must be positive.")
    has_bottom = (profile[0, 0] <= rm.eps)
    has_top = (profile[-1, 0] <= rm.eps)
    profile_core = profile[1:] if has_bottom else profile
    profile_core = profile_core[:-1] if has_top else profile_core
    r_core = profile_core[:, 0]
    z_core = profile_core[:, 1]
    n_core = len(r_core)
    # angles
    theta = rm.linspace(0, 2 * rm.pi, segments, endpoint=False)
    cos_t = rm.cos(theta)
    sin_t = rm.sin(theta)
    # verts
    r_mat = rm.tile(r_core, (segments, 1))
    z_mat = rm.tile(z_core, (segments, 1))
    x = r_mat * cos_t[:, None]
    y = r_mat * sin_t[:, None]
    verts = rm.stack([x, y, z_mat], axis=2).reshape(-1, 3)
    # side wall faces
    idx = rm.arange(segments * n_core, dtype=rm.uint32).reshape(segments, n_core)
    idx_next = rm.roll(idx, -1, axis=0)
    # tri1 = (i,j), (i+1,j), (i,j+1)
    tri1 = rm.stack([idx[:, :-1], idx_next[:, :-1], idx[:, 1:]], axis=2).reshape(-1, 3)
    # tri2 = (i,j+1), (i+1,j), (i+1,j+1)
    tri2 = rm.stack([idx[:, 1:], idx_next[:, :-1], idx_next[:, 1:]], axis=2).reshape(-1, 3)
    faces_list = [tri1, tri2]
    extra_verts = []
    # bottom cap
    ring_bottom_idx = idx[:, 0]  # shape (seg,)
    verts_bottom = verts[ring_bottom_idx]
    if has_bottom:
        center_bottom = rm.array([0.0, 0.0, profile[0, 1]], dtype=rm.float32)
    else:
        center_bottom = verts_bottom.mean(axis=0)
    center_bottom_idx = len(verts)  # new idx for center vertex
    extra_verts.append(center_bottom)
    tri_bottom = rm.stack([rm.full(segments, center_bottom_idx, dtype=rm.uint32), rm.roll(ring_bottom_idx, -1),
                           ring_bottom_idx], axis=1)
    faces_list.append(tri_bottom)
    # top cap
    ring_top_idx = idx[:, -1]  # shape (seg,)
    verts_top = verts[ring_top_idx]
    if has_top:
        center_top = rm.array([0.0, 0.0, profile[-1, 1]], dtype=rm.float32)
    else:
        center_top = verts_top.mean(axis=0)
    center_top_idx = len(verts) + len(extra_verts)  # new idx for center vertex
    extra_verts.append(center_top)
    tri_top = rm.stack([rm.full(segments, center_top_idx, dtype=rm.uint32), ring_top_idx,
                        rm.roll(ring_top_idx, -1)], axis=1)
    faces_list.append(tri_top)
    verts = rm.vstack([verts, rm.vstack(extra_verts)])
    faces = rm.concatenate(faces_list, axis=0)
    return (verts, faces)

def subdivide_once(verts, faces):
    edges = rm.sort(rm.stack([faces[:, [0, 1]],
                              faces[:, [1, 2]],
                              faces[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
    # unique edges and their midpoints
    unique, inverse = rm.unique(edges, axis=0, return_inverse=True)
    mid = verts[unique].mean(axis=1)
    mid_idx = inverse.reshape(-1, 3) + len(verts)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m0, m1, m2 = mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]
    new_faces = rm.vstack([rm.column_stack([v0, m0, m2]),
                           rm.column_stack([v1, m1, m0]),
                           rm.column_stack([v2, m2, m1]),
                           rm.column_stack([m0, m1, m2])])
    new_verts = rm.vstack([verts, mid])
    new_verts /= rm.linalg.norm(new_verts, axis=1, keepdims=True)
    return new_verts, new_faces

def icosahedron():
    """Generate vertices and faces of a unit icosahedron."""
    t = (1 + rm.sqrt(5)) / 2
    verts = rm.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ], dtype=rm.float32)
    faces = rm.array([
        [0,11,5],  [0,5,1],   [0,1,7],   [0,7,10],  [0,10,11],
        [1,5,9],   [5,11,4],  [11,10,2], [10,7,6],  [7,1,8],
        [3,9,4],   [3,4,2],   [3,2,6],   [3,6,8],   [3,8,9],
        [4,9,5],   [2,4,11],  [6,2,10],  [8,6,7],   [9,8,1]
    ], dtype=rm.uint32)
    # normalize (unit sphere)
    verts /= rm.linalg.norm(verts, axis=1, keepdims=True)
    return verts, faces