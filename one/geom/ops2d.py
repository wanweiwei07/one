import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def pts_in_polygon2d(pts, poly_pts):
    # pts: (M,2), poly_pts: (N,2)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1: # single point
        pts = pts[None, :]
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    x1 = poly_pts[:, 0][None, :]
    y1 = poly_pts[:, 1][None, :]
    x2 = np.roll(poly_pts[:, 0], -1)[None, :]
    y2 = np.roll(poly_pts[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
    crossings = (x < x_int) & cond
    return np.sum(crossings, axis=1) % 2 == 1


def mindist_to_linesegs2d(pt, line_segs):
    """pt: (2,), line_segs: array-like (M,2,2)
    return: (min_dist, closest_point)"""
    a = line_segs[:, 0, :]  # (M,2)
    b = line_segs[:, 1, :]  # (M,2)
    ab = b - a
    ap = pt - a
    denom = np.sum(ab * ab, axis=1) + 1e-12
    t = np.sum(ap * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)[:, None]
    q = a + t * ab  # (M,2)
    diff = pt - q
    d2 = np.sum(diff * diff, axis=1)
    idx = np.argmin(d2)
    return float(np.sqrt(d2[idx])), q[idx]


def extract_boundary(fs_sub):
    # fs_sub: (K,3) fs (a subset of geom.fs)
    # return list of vids forming boundary polygon
    edges = np.stack([
        np.sort(fs_sub[:, [0, 1]], axis=1),
        np.sort(fs_sub[:, [1, 2]], axis=1),
        np.sort(fs_sub[:, [2, 0]], axis=1)
    ], axis=1).reshape(-1, 2)
    # find boundary edges
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if len(boundary_edges) == 0:
        return []  # if no boundary edges
    # check if boundary edges form a single loop
    nodes = np.unique(boundary_edges)
    node_id = {n: i for i, n in enumerate(nodes)}
    row = np.array(
        [node_id[a]
         for a in boundary_edges[:, 0]])
    col = np.array(
        [node_id[b]
         for b in boundary_edges[:, 1]])
    data = np.ones(len(row) * 2, dtype=np.uint8)
    rows = np.concatenate([row, col])
    cols = np.concatenate([col, row])
    A = coo_matrix((data, (rows, cols)),
                   shape=(len(nodes), len(nodes)))
    n_comp, labels = connected_components(A, directed=False)
    if n_comp > 1:
        print(f"multiple boundary loops = {n_comp}")
        return []
    # adjacency list
    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # traverse to form polygon (list of vids)
    start = boundary_edges[0, 0]
    poly_vids = [start]
    prev = None
    cur = start
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        poly_vids.append(nxt)
        prev, cur = cur, nxt
    return poly_vids
