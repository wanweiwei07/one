"""Antipodal grasp planning: 2-point opposing pinch (force-closure) grasps
for a parallel-jaw gripper. ``antipodal``/``antipodal_iter`` sample the
target surface for antipodal contact pairs, align the jaw, and reject
gripper-vs-target collisions. See also polypodal (N-point) and monocontact
(single-contact / suction)."""
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.geometry_ops as osgop
import one.collider.cpu_simd as occs
import one.grasp._common as ogc


def build_grasp_rotmat_batch(ray_dirs, open_dir):
    """
    ray_dirs: (N,3) unit vectors
    open_dir: gripper opening direction in TCP local frame -- (3,) shared by all
        candidates, or (N,3) one per candidate (e.g. a closure-dependent pinch).
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
    rot_base = np.stack([x, y, z], axis=2).astype(np.float32)
    open_dir = np.asarray(open_dir, dtype=np.float32)
    if np.linalg.norm(open_dir) < oum.eps:
        raise ValueError('open_dir must be non-zero')
    offset = oum.rotmat_between_vecs(
        open_dir, ouc.StandardAxis.Y).astype(np.float32)
    if offset.ndim == 2:                       # shared open_dir -> single offset
        return rot_base @ offset
    return np.einsum('nij,njk->nik', rot_base, offset)   # per-candidate


def _antipodal_candidates(
        tgt_vs, tgt_fs, tgt_fns,
        density, normal_tol_deg,
        roll_step_deg, clearance):
    normal_cos_th = np.cos(np.deg2rad(normal_tol_deg))
    roll_step = np.deg2rad(roll_step_deg)
    n_samples = osgop.sample_count_from_area(tgt_vs, tgt_fs, density)
    pts, nrms, _ = osgop.sample_surface(tgt_vs, tgt_fs, n_samples)
    v0 = tgt_vs[tgt_fs[:, 0]]
    v1 = tgt_vs[tgt_fs[:, 1]]
    v2 = tgt_vs[tgt_fs[:, 2]]
    origins = pts
    directions = -nrms
    hit_t, hit_id = osgop.ray_triangles_batch_far(
        origins, directions, v0, v1, v2, eps=float(oum.eps))
    valid = hit_id >= 0
    if not np.any(valid):
        return None
    origins_v = origins[valid]
    directions_v = directions[valid]
    hit_t_v = hit_t[valid]
    hit_id_v = hit_id[valid]
    hit_pos = origins_v + hit_t_v[:, None] * directions_v
    hit_n = tgt_fns[hit_id_v]
    p_all = origins_v
    n_all = nrms[valid]
    q_all = hit_pos
    nq_all = hit_n
    dot_nn = np.einsum("ij,ij->i", n_all, -nq_all)
    n_norm = np.linalg.norm(n_all, axis=1) + oum.eps
    nq_norm = np.linalg.norm(nq_all, axis=1) + oum.eps
    cos_vals = dot_nn / (n_norm * nq_norm)
    jaw_width = np.linalg.norm(q_all - p_all, axis=1) + 2 * clearance
    center = (p_all + q_all) * 0.5
    return center, jaw_width, n_all, nq_all, cos_vals, normal_cos_th, roll_step


def antipodal_iter(gripper, target_sobj,
                   density=0.02, normal_tol_deg=20,
                   roll_step_deg=30, clearance=0.002,
                   score_weights=(0.7, 0.3),
                   exclude_regions=None, pre_open=0.5):
    """
    Generator: yields (pose, pre_pose, jaw_width, score, collided).
    :param gripper: gripper instance
    :param target_sobj: target object to grasp
    :param density: surface sampling density
    :param normal_tol_deg: normal tolerance in degrees
    :param roll_step_deg: roll angle step in degrees
    :param clearance: additional clearance for jaw width
    :param score_weights: (normal_align_weight, jaw_close_weight)
    :param exclude_regions: optional list of convex regions (each a
        list of (point, normal) half-space constraints) carved out of
        the surface before contact-pair sampling. The original target
        mesh is still used for gripper-vs-target collision checks, so
        grasps near a clipped feature will still be rejected if the
        gripper would collide with that feature.
    :param pre_open: how far OPEN the jaw is at the pre-grasp pose, as a
        fraction in [0, 1] of the room between the grasp width and the
        max opening (pre_jw = jw + pre_open*(jaw_max - jw)). 0 keeps it
        at the grasp width; the default 0.5 opens half-way so the
        collision check reflects the wider hand swept in on approach.
    Uses gripper.contact_pattern to confirm this is a single-contact
        model and to compensate jaw width for contact depth along the
        gripper opening axis. TCP is aligned to the two-contact midpoint.
    :return: yields (pose, pre_pose, jaw_width, score, collided), where
        pose: 4x4 world transform of the GRASP CENTER (the grasp_center tcp
            frame) when closed on the object -- its origin is the contact-pair
            midpoint, its +z is the approach axis. This is exactly the
            (tgt_pos, tgt_rotmat) the gripper's grip_at expects.
        pre_pose: 4x4 world transform of the pre-grasp pose, pose retreated
            along the approach axis (-pose[:3, 2]); its collision check uses
            the jaw opened part-way (see pre_open), not the grasp width.
        jaw_width: jaw opening for this grasp.
        score: score_weights-weighted normal-alignment + jaw-centering.
        collided: True if the gripper collides with the target at pose or
            pre_pose (both poses are collision-checked).
    """
    gripper = gripper.clone()
    # Plan in the target's LOCAL (zero-pose) frame. Clone the target and zero
    # its pose so the contact sampling (local geom) and the gripper-vs-target
    # collision check (CollisionBatch uses target.tf) live in the SAME frame.
    # The returned grasps are therefore in the target's local frame; the caller
    # maps them onto the placed object (e.g. target_sobj.wd_tf @ pose).
    target_sobj = target_sobj.clone()
    target_sobj.set_pos_rotmat(
        pos=np.zeros(3, dtype=np.float32), rotmat=np.eye(3, dtype=np.float32))
    tgt_vs, tgt_fs, tgt_fns = occs.cols_to_vffns(
        target_sobj.collisions)
    if exclude_regions:
        tgt_vs, tgt_fs = osgop.clip_mesh(
            tgt_vs, tgt_fs, exclude_regions)
        if len(tgt_fs) == 0:
            return []
        v0 = tgt_vs[tgt_fs[:, 0]]
        v1 = tgt_vs[tgt_fs[:, 1]]
        v2 = tgt_vs[tgt_fs[:, 2]]
        tgt_fns = np.cross(v1 - v0, v2 - v0)
        tgt_fns = tgt_fns / (
            np.linalg.norm(tgt_fns, axis=1, keepdims=True) + oum.eps)
        tgt_fns = tgt_fns.astype(np.float32)
    cand = _antipodal_candidates(
        tgt_vs, tgt_fs, tgt_fns, density,
        normal_tol_deg, roll_step_deg, clearance)
    if cand is None:
        return []
    (center, jaw_width, n_all, nq_all,
     cos_vals, normal_cos_th, roll_step) = cand
    contact_pattern = np.asarray(gripper.contact_pattern, dtype=np.float32)
    if contact_pattern.ndim != 2 or contact_pattern.shape != (1, 3):
        raise ValueError(
            'antipodal requires gripper.contact_pattern to be (1, 3)'
        )
    open_dir = gripper.open_dir / (np.linalg.norm(gripper.open_dir) + oum.eps)
    contact_depth = abs(float(contact_pattern[0] @ open_dir))
    jaw_width = jaw_width - 2.0 * contact_depth
    jaw_min, jaw_max = gripper.jaw_range
    jaw_mid = 0.5 * (jaw_min + jaw_max)
    jaw_span = jaw_max - jaw_min + oum.eps
    mask = ((cos_vals >= normal_cos_th) &
            (jaw_width >= jaw_min) &
            (jaw_width <= jaw_max))
    if not np.any(mask):
        return []
    center_sel = center[mask]
    jaw_sel = jaw_width[mask]
    n_sel = n_all[mask]
    nq_sel = nq_all[mask]
    ray_dirs = -n_sel
    # per-candidate opening axis when the gripper provides one (e.g. a curling
    # pinch whose pad-opposition axis depends on the closure width); else the
    # single constant open_dir (a true parallel jaw).
    if hasattr(gripper, 'open_dir_at'):
        open_dir_sel = np.asarray(gripper.open_dir_at(jaw_sel), dtype=np.float32)
    else:
        open_dir_sel = open_dir
    rot_base = build_grasp_rotmat_batch(ray_dirs, open_dir_sel)
    angles = np.arange(0.0, 2 * np.pi, roll_step)
    if open_dir_sel.ndim == 2:
        roll_axes = np.einsum('nij,nj->ni', rot_base, open_dir_sel)
    else:
        roll_axes = np.einsum('nij,j->ni', rot_base, open_dir_sel)
    roll_rots = oum.rotmat_from_axangle(roll_axes[:, None, :], angles[None, :])
    rot_all = roll_rots @ rot_base[:, None, :, :]
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
    # prepare collision batch
    detector, batch = ogc.build_ee_target_detector(gripper, target_sobj)
    # retreat distance
    tcp_len = np.linalg.norm(gripper.tcp('grasp_center').loc_tf[:3, 3])
    retreat_dist = 0.5 * tcp_len
    for pose, jw, sc in zip(pose_all, jaw_all, score_all):
        collided = False
        # check grasp pose
        gripper.grip_at(pose[:3, 3], pose[:3, :3], jw)
        results = detector.detect_collision_batch(batch)
        if results is not None:
            collided = True
        # check pre-grasp pose, with the jaw OPENED PART-WAY toward its max
        # (pre_open in [0, 1]): on approach the fingers should clear the object,
        # not already be closed to the grasp width. 0 -> same as grasp, 1 -> full
        # open. The check thus reflects the wider hand actually swept in.
        pre_jw = jw + pre_open * (jaw_max - jw)
        pre_pos = pose[:3, 3] - retreat_dist * pose[:3, 2]
        pre_pose = pose.copy()
        pre_pose[:3, 3] = pre_pos
        gripper.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], pre_jw)
        results = detector.detect_collision_batch(batch)
        if results is not None:
            collided = True
        yield pose, pre_pose, jw, float(sc), collided


def antipodal(gripper, target_sobj,
              density=0.02, normal_tol_deg=20,
              roll_step_deg=30, clearance=0.002,
              max_grasps=50, score_weights=(0.7, 0.3),
              exclude_regions=None, pre_open=0.5):
    """
    Collects non-colliding grasps only.
    :param gripper: gripper instance
    :param target_sobj: target object to grasp
    :param density: surface sampling density
    :param normal_tol_deg: normal tolerance in degrees
    :param roll_step_deg: roll angle step in degrees
    :param clearance: additional clearance for jaw width
    :param max_grasps: maximum number of grasps to return
    :param score_weights: (normal_align_weight, jaw_close_weight)
    :param exclude_regions: optional list of convex regions to carve
        out of the contact-sampling surface (forwarded to
        antipodal_iter). The full mesh is still used for collision
        checks.
    :param pre_open: jaw opening at the pre-grasp pose as a fraction of
        the room to max (default 0.5 = half-open); forwarded to
        antipodal_iter.
    :return: list of (pose, pre_pose, jaw_width, score) for the
        collision-free grasps, best score first. pose is the 4x4 world
        transform of the GRASP CENTER (grasp_center tcp frame, origin at the
        contact-pair midpoint, +z = approach axis) -- pass pose[:3, 3],
        pose[:3, :3] straight to the gripper's grip_at. pre_pose is the
        pre-grasp pose (pose retreated along the approach axis). See
        antipodal_iter for the full field description.
    """
    results = []
    for pose, pre_pose, jw, sc, collided in antipodal_iter(
            gripper, target_sobj,
            density, normal_tol_deg, roll_step_deg,
            clearance, score_weights,
            exclude_regions=exclude_regions, pre_open=pre_open):
        if not collided:
            results.append((pose, pre_pose, jw, float(sc)))
        if (max_grasps is not None and
                len(results) >= max_grasps):
            break
    return results
