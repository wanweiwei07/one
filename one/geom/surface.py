import numpy as np


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


def segment_surface(geometry, normal_tol_deg=5.0):
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
    facets = []
    for c in range(n_comp):
        fids_c = np.where(labels == c)[0]
        if fids_c.size == 0:
            continue
        facets.append(fids_c)
    return tuple(facets)