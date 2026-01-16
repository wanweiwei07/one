import numpy as np
import one.utils.math as oum
import one.scene.geometry_operation as osgop


def _triangle_areas(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _sample_count_from_area(verts, faces, density):
    areas = _triangle_areas(verts, faces)
    area_total = float(np.sum(areas))
    n = int(np.ceil(area_total / (density * density)))
    return max(n, 1)


def _ray_triangles_batch_far(origins, directions, v0, v1, v2, eps=1e-6):
    """Moller-Trumbore ray-triangle intersection (batch version)."""
    o = origins[:, None, :]
    d = directions[:, None, :]
    v0 = v0[None, :, :]
    v1 = v1[None, :, :]
    v2 = v2[None, :, :]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(d, e2)
    a = np.sum(e1 * h, axis=2)
    mask = np.abs(a) > eps
    f = np.zeros_like(a)
    f[mask] = 1.0 / a[mask]
    s = o - v0
    u = f * np.sum(s * h, axis=2)
    mask &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(s, e1)
    v = f * np.sum(d * q, axis=2)
    mask &= (v >= 0.0) & (u + v <= 1.0)
    t = f * np.sum(e2 * q, axis=2)
    mask &= (t > eps)
    t_masked = np.where(mask, t, -np.inf)
    hit_t = np.max(t_masked, axis=1)
    hit_id = np.argmax(t_masked, axis=1)
    no_hit = np.isneginf(hit_t)
    hit_t = np.where(no_hit, -1.0, hit_t)
    hit_id = np.where(no_hit, -1, hit_id)
    return hit_t, hit_id


def _merge_collision_meshes(scene_obj):
    verts_all = []
    faces_all = []
    normals_all = []
    offset = 0
    for col in scene_obj.collisions:
        geom = col.geometry
        if geom.faces is None:
            continue
        verts = geom.verts
        faces = geom.faces
        fn = geom.face_normals
        rot = col.rotmat
        pos = col.pos
        verts_s = (rot @ verts.T).T + pos
        fn_s = (rot @ fn.T).T
        verts_all.append(verts_s)
        normals_all.append(fn_s)
        faces_all.append(faces + offset)
        offset += verts.shape[0]
    if not verts_all:
        return None
    verts_all = np.vstack(verts_all)
    faces_all = np.vstack(faces_all)
    normals_all = np.vstack(normals_all)
    return verts_all, faces_all, normals_all


def build_grasp_rotmat_batch(ray_dirs):
    """
    ray_dirs: (N,3) unit vectors (Y axis for each grasp)
    returns: rotmats (N,3,3)
    """
    y = ray_dirs / (np.linalg.norm(ray_dirs, axis=1, keepdims=True) + oum.eps)
    ref1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ref2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot1 = np.abs(y @ ref1)
    dot2 = np.abs(y @ ref2)
    use_ref1 = dot1 < dot2
    ref = np.where(use_ref1[:, None], ref1, ref2)
    z = ref - (np.sum(ref * y, axis=1, keepdims=True) * y)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + oum.eps)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + oum.eps)
    return np.stack([x, y, z], axis=2).astype(np.float32)


def _rotmat_about_axis_batch_vec(axes, angles):
    """
    axes: (N,3) unit vectors
    angles: (K,)
    returns: (N,K,3,3)
    """
    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + oum.eps)
    ax = axes[:, None, :]  # (N,1,3)
    x = ax[..., 0]
    y = ax[..., 1]
    z = ax[..., 2]
    c = np.cos(angles)[None, :]  # (1,K)
    s = np.sin(angles)[None, :]
    C = 1.0 - c
    rot = np.zeros((axes.shape[0], angles.shape[0], 3, 3), dtype=np.float32)
    rot[:, :, 0, 0] = c + x * x * C
    rot[:, :, 0, 1] = x * y * C - z * s
    rot[:, :, 0, 2] = x * z * C + y * s
    rot[:, :, 1, 0] = y * x * C + z * s
    rot[:, :, 1, 1] = c + y * y * C
    rot[:, :, 1, 2] = y * z * C - x * s
    rot[:, :, 2, 0] = z * x * C - y * s
    rot[:, :, 2, 1] = z * y * C + x * s
    rot[:, :, 2, 2] = c + z * z * C
    return rot


def _antipodal_candidates(
        scene_obj, density, normal_tol_deg,
        roll_step_deg, clearance):
    merged = _merge_collision_meshes(scene_obj)
    if merged is None:
        return None
    verts, faces, face_normals = merged
    normal_cos_th = np.cos(np.deg2rad(180.0 - normal_tol_deg))
    roll_step = np.deg2rad(roll_step_deg)
    n_samples = _sample_count_from_area(verts, faces, density)
    pts, nrms, _ = osgop.sample_surface(verts, faces, n_samples)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    origins = pts
    directions = -nrms
    hit_t, hit_id = _ray_triangles_batch_far(origins, directions, v0, v1, v2, eps=float(oum.eps))
    valid = hit_id >= 0
    if not np.any(valid):
        return None
    origins_v = origins[valid]
    directions_v = directions[valid]
    hit_t_v = hit_t[valid]
    hit_id_v = hit_id[valid]
    hit_pos = origins_v + hit_t_v[:, None] * directions_v
    hit_n = face_normals[hit_id_v]
    p_all = origins_v
    n_all = nrms[valid]
    q_all = hit_pos
    nq_all = hit_n
    dot_nn = np.einsum("ij,ij->i", n_all, -nq_all)
    n_norm = np.linalg.norm(n_all, axis=1) + oum.eps
    nq_norm = np.linalg.norm(nq_all, axis=1) + oum.eps
    cos_vals = dot_nn / (n_norm * nq_norm)
    jaw_width = np.linalg.norm(q_all - p_all, axis=1) + 15 * clearance
    center = (p_all + q_all) * 0.5
    return center, jaw_width, n_all, nq_all, cos_vals, normal_cos_th, roll_step


def antipodal_iter(scene_obj, gripper, mj_collider,
                   density=0.02, normal_tol_deg=20,
                   roll_step_deg=30, clearance=0.002,
                   score_weights=(0.7, 0.3)):
    """
    Generator: yields (pose_tf, jaw_width, score, collided)
    Also collects non-colliding grasps and returns them on StopIteration.
    """
    cand = _antipodal_candidates(
        scene_obj, density, normal_tol_deg,
        roll_step_deg, clearance)
    if cand is None:
        return []
    center, jaw_width, n_all, nq_all, cos_vals, normal_cos_th, roll_step = cand
    jaw_min, jaw_max = gripper.jaw_range
    jaw_mid = 0.5 * (jaw_min + jaw_max)
    jaw_span = jaw_max - jaw_min + oum.eps
    mask = (cos_vals >= normal_cos_th) & (jaw_width >= jaw_min) & (jaw_width <= jaw_max)
    if not np.any(mask):
        return []
    center_sel = center[mask]
    jaw_sel = jaw_width[mask]
    n_sel = n_all[mask]
    nq_sel = nq_all[mask]
    ray_dirs = -n_sel
    rot_base = build_grasp_rotmat_batch(ray_dirs)  # (N,3,3)
    angles = np.arange(0.0, 2 * np.pi, roll_step)
    roll_rots = _rotmat_about_axis_batch_vec(rot_base[:, :, 1], angles)  # (N,K,3,3)
    rot_all = roll_rots @ rot_base[:, None, :, :]  # (N,K,3,3)
    pose_tf = np.tile(np.eye(4, dtype=np.float32),
                      (rot_all.shape[0], rot_all.shape[1], 1, 1))
    pose_tf[:, :, :3, :3] = rot_all
    pose_tf[:, :, :3, 3] = center_sel[:, None, :]
    pose_all = pose_tf.reshape(-1, 4, 4)
    jaw_all = np.repeat(jaw_sel, len(angles))
    normal_align = (1.0 + np.einsum("ij,ij->i", n_sel, -nq_sel) /
                    (np.linalg.norm(n_sel, axis=1) *
                     np.linalg.norm(nq_sel, axis=1) + oum.eps)) * 0.5
    jaw_close = 1.0 - np.abs(jaw_sel - jaw_mid) / jaw_span
    score = score_weights[0] * normal_align + score_weights[1] * jaw_close
    score_all = np.repeat(score, len(angles))
    order = np.argsort(score_all)[::-1]
    pose_all = pose_all[order]
    jaw_all = jaw_all[order]
    score_all = score_all[order]
    for pose, jw, sc in zip(pose_all, jaw_all, score_all):
        gripper.grip_at(pose[:3, 3], pose[:3, :3], jw)
        collided = mj_collider.is_collided(gripper.qs)
        yield pose, jw, float(sc), collided


def antipodal(scene_obj, gripper, mj_collider,
              density=0.02, normal_tol_deg=20,
              roll_step_deg=30, clearance=0.002,
              max_grasps=50, score_weights=(0.7, 0.3)):
    """  Collects non-colliding grasps only.   """
    results = []
    for pose, jw, sc, collided in antipodal_iter(
            scene_obj, gripper, mj_collider,
            density, normal_tol_deg, roll_step_deg,
            clearance, score_weights):
        if not collided:
            results.append((pose.copy(), jw, float(sc)))
            if max_grasps is not None and len(results) >= max_grasps:
                break
    return results
