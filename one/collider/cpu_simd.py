import numpy as np
import one.collider.collision_batch as occb


class CPUDetector():
    def __init__(self, eps=1e-9, max_points=200):
        self.eps = eps
        self.max_points = max_points

    def detect_collision(self, vs_a, fs_a, tf_a,
                         vs_b, fs_b, tf_b):
        """
        detect collision between two sets of triangles using CPU
        """
        # Transform
        vs_a_tfed = (vs_a @ tf_a[:3, :3].T) + tf_a[:3, 3]
        vs_b_tfed = (vs_b @ tf_b[:3, :3].T) + tf_b[:3, 3]
        tris_a = vs_a_tfed[fs_a]
        tris_b = vs_b_tfed[fs_b]
        # Compute AABBs and check for overlap
        min_a = tris_a.min(axis=(0, 1))
        max_a = tris_a.max(axis=(0, 1))
        min_b = tris_b.min(axis=(0, 1))
        max_b = tris_b.max(axis=(0, 1))
        if not aabb_intersect(min_a, max_a, min_b, max_b):
            return None
        tri_min_a = tris_a.min(axis=1)
        tri_max_a = tris_a.max(axis=1)
        tri_min_b = tris_b.min(axis=1)
        tri_max_b = tris_b.max(axis=1)
        overlap_mask = ((tri_min_a[:, None, :] <= tri_max_b[None, :, :]) &
                        (tri_max_a[:, None, :] >= tri_min_b[None, :, :])).all(axis=-1)
        pair_indices = np.argwhere(overlap_mask)
        if len(pair_indices) == 0:
            return None
        # Detailed triangle-triangle intersection test
        tris_a_pairs = tris_a[pair_indices[:, 0]]
        tris_b_pairs = tris_b[pair_indices[:, 1]]
        result = self._tripair_planeprojection_filter(
            tris_a_pairs, tris_b_pairs)
        if result is None:
            return None
        tris_pairs = result
        fine_result = self._tripair_fine_filter(
            tris_pairs[0], tris_pairs[1])
        if fine_result is None:
            return None
        else:
            return fine_result

    def detect_collision_batch(self, batch):
        if batch is None or batch.pairs is None or len(batch.pairs) == 0:
            return None
        # update transforms
        batch.update_transforms()
        # whole-body AABB
        wd_min, wd_max = occb.compute_wd_aabb_batch(
            batch.aabb_mins, batch.aabb_maxs, batch.tfs)
        ia = batch.pairs[:, 0]
        ib = batch.pairs[:, 1]
        overlap = (wd_min[ia] <= wd_max[ib]).all(axis=1) & \
                  (wd_max[ia] >= wd_min[ib]).all(axis=1)
        cand = np.where(overlap)[0]
        if cand.size == 0:
            return None
        points_out = []
        pair_ids_out = []
        desc = batch.geom_descs
        for id in cand:
            i, j = batch.pairs[id]
            v_off_a, v_cnt_a, f_off_a, f_cnt_a = desc[i]
            v_off_b, v_cnt_b, f_off_b, f_cnt_b = desc[j]
            if f_cnt_a == 0 or f_cnt_b == 0:
                continue
            vs_a = batch.vss[v_off_a:v_off_a + v_cnt_a]
            fs_a = batch.fss[f_off_a:f_off_a + f_cnt_a] - v_off_a
            tf_a = batch.tfs[i]
            vs_b = batch.vss[v_off_b:v_off_b + v_cnt_b]
            fs_b = batch.fss[f_off_b:f_off_b + f_cnt_b] - v_off_b
            tf_b = batch.tfs[j]
            pts = self.detect_collision(
                vs_a, fs_a, tf_a, vs_b, fs_b, tf_b)
            if pts is None:
                continue
            for p in pts:
                points_out.append(p)
                pair_ids_out.append(id)
                if len(points_out) >= self.max_points:
                    break
            if len(points_out) >= self.max_points:
                break
        if not points_out:
            return None
        return (np.asarray(points_out, np.float32),
                np.asarray(pair_ids_out, np.uint32))

    def _tripair_planeprojection_filter(self, tris_a, tris_b):
        """
        Find first intersecting triangle pair.
        :param tris_a, tris_b: (N,3,3)
        :return: (hit_found, (ia, ib)) or (False, None)
        """
        normals_a, offsets_a = compute_triangle_planes(tris_a)
        normals_b, offsets_b = compute_triangle_planes(tris_b)
        dist_to_plane_a = (np.einsum("nij,nj->ni", tris_b, normals_a) +
                           offsets_a[:, None])
        dist_to_plane_b = (np.einsum("nij,nj->ni", tris_a, normals_b) +
                           offsets_b[:, None])
        separated_by_a = ((dist_to_plane_a > self.eps).all(axis=1) |
                          (dist_to_plane_a < -self.eps).all(axis=1))
        separated_by_b = ((dist_to_plane_b > self.eps).all(axis=1) |
                          (dist_to_plane_b < -self.eps).all(axis=1))
        candidate_mask = ~(separated_by_a | separated_by_b)
        if not np.any(candidate_mask):
            return None
        idx = np.where(candidate_mask)[0]
        tris_a = tris_a[idx]
        tris_b = tris_b[idx]
        normals_a = normals_a[idx]
        normals_b = normals_b[idx]
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
            return None
        tris_a_hit = tris_a[hit_local]
        tris_b_hit = tris_b[hit_local]
        return (tris_a_hit, tris_b_hit)

    def _tripair_fine_filter(self, tris_a, tris_b):
        nA = np.cross(tris_a[:, 1] - tris_a[:, 0], tris_a[:, 2] - tris_a[:, 0])
        nB = np.cross(tris_b[:, 1] - tris_b[:, 0], tris_b[:, 2] - tris_b[:, 0])
        nA /= (np.linalg.norm(nA, axis=1, keepdims=True) + self.eps)
        nB /= (np.linalg.norm(nB, axis=1, keepdims=True) + self.eps)
        dA = -np.einsum("ij,ij->i", nA, tris_a[:, 0])
        dB = -np.einsum("ij,ij->i", nB, tris_b[:, 0])
        dir_vec = np.cross(nA, nB)
        dir_norm = np.linalg.norm(dir_vec, axis=1)
        valid = dir_norm >= 1e-4
        p0 = np.cross((dB[:, None] * nA - dA[:, None] * nB),
                      dir_vec) / (dir_norm[:, None] ** 2 + self.eps)
        dir_vec = dir_vec / (dir_norm[:, None] + self.eps)
        proj_a = np.einsum("nij,nj->ni", tris_a, dir_vec)
        proj_b = np.einsum("nij,nj->ni", tris_b, dir_vec)
        t0 = np.maximum(proj_a.min(axis=1), proj_b.min(axis=1))
        t1 = np.minimum(proj_a.max(axis=1), proj_b.max(axis=1))
        valid &= (t0 < t1)
        points = p0 + dir_vec * ((t0 + t1) * 0.5)[:, None]
        inside_a = point_in_tri_batch(points, tris_a)
        inside_b = point_in_tri_batch(points, tris_b)
        valid &= inside_a & inside_b
        if np.any(valid):
            return points[valid]
        else:
            return None

def create_detector(eps=1e-9, max_points=200):
    return CPUDetector(eps=eps, max_points=max_points)

def build_batch(items, pairs):
    return occb.CollisionBatch(items, pairs)

def cols_to_vfs(cols):
    if not cols:
        return None
    verts_all = []
    faces_all = []
    offset = 0
    for col in cols:
        geom = col.geometry
        tf = col.tf
        rot = tf[:3, :3]
        pos = tf[:3, 3]
        verts = (rot @ geom.vs.T).T + pos
        faces = geom.fs + offset
        verts_all.append(verts)
        faces_all.append(faces)
        offset += verts.shape[0]
    verts_all = np.vstack(verts_all).astype(
        np.float32, copy=False)
    faces_all = np.vstack(faces_all).astype(
        np.int32, copy=False)
    return verts_all, faces_all


def cols_to_vffns(cols):
    if not cols:
        return None
    verts_all = []
    faces_all = []
    fn_all = []
    offset = 0
    for col in cols:
        geom = col.geometry
        tf = col.tf
        rot = tf[:3, :3]
        pos = tf[:3, 3]
        verts = (rot @ geom.vs.T).T + pos
        faces = geom.fs + offset
        fn = (rot @ geom.fns.T).T
        verts_all.append(verts)
        faces_all.append(faces)
        fn_all.append(fn)
        offset += verts.shape[0]
    verts_all = np.vstack(verts_all).astype(
        np.float32, copy=False)
    faces_all = np.vstack(faces_all).astype(
        np.int32, copy=False)
    fn_all = np.vstack(fn_all).astype(
        np.float32, copy=False)
    return verts_all, faces_all, fn_all


def cols_to_tris(cols):
    out = cols_to_vfs(cols)
    if out is None:
        return None
    verts, faces = out
    return verts[faces]

def compute_aabb(tris):
    mins = tris.min(axis=(0, 1))
    maxs = tris.max(axis=(0, 1))
    return mins, maxs


def aabb_intersect(min_a, max_a, min_b, max_b):
    return np.all(min_a <= max_b) and np.all(max_a >= min_b)


def compute_triangle_planes(tris):
    """
    plane through origin: n dot x = 0
    plane through point p: n dot x + d = 0, d = -n dot p
    :param tris: (N,3,3)
    :return: normals: (N,3), offsets: (N,)
    """
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(edge1, edge2)
    offsets = -np.einsum(  # -n dot p
        "ij,ij->i", normals, tris[:, 0])
    return normals, offsets


def point_in_tri_batch(pts, tris, eps=1e-9):
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
