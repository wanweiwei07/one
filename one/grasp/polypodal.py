"""Polypodal: rigid N-point contact pattern matching against a mesh
surface.

A `pattern` is a tuple of N 3D points expressed in the TCP frame when
the gripper is closed. Each pattern[k] is a contact pad's position
relative to the TCP origin. All N points are assumed coplanar; the
plane normal aligns with the gripper's open axis.

Pipeline (highest-level entry point: `polypodal`):
    sample_pattern   -> N-point placements on the front face
    pair_pattern     -> ray-cast each placement to the back face
    _hand_poses_from_pair -> Procrustes register pattern to contacts
    _check_grasp_collision -> reject gripper-vs-target collisions

Outputs are (pose_4x4, jaw_width) tuples, with TCP world position at
the midpoint between the front and back contact centroids.
"""
import numpy as np
from scipy.spatial import cKDTree

import one.collider.cpu_simd as occs
import one.collider.gpu_simd_batch as ocgcb
import one.scene.geometry_ops as osgop
from one.grasp.antipodal import _ray_triangles_batch_far


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _hand_poses_from_pair(pattern, front_pts, back_pts,
                          open_dir=None, clearance=0.0003):
    """Compute TCP target-frame pose candidates for one (front, back)
    pair, assuming `pattern` is already in the TCP frame.

    Rotation: Procrustes register the pattern to front and back
        contacts separately; an augmented anchor along the pattern
        normal pinned at the opposite centroid resolves the coplanar
        rotation ambiguity.
    Translation: TCP is placed at the opening-axis midpoint, preserving
        only the pattern centroid's tangential offset.
    Jaw width: front/back centroid distance plus per-side clearance,
        minus the two contact offsets along the gripper opening axis.

    Returns a list of (pose 4x4, jaw_width).
    """
    pattern = np.asarray(pattern, dtype=np.float32)
    front = np.asarray(front_pts, dtype=np.float32)
    back = np.asarray(back_pts, dtype=np.float32)
    p_c = pattern.mean(axis=0)
    f_c = front.mean(axis=0)
    b_c = back.mean(axis=0)
    contact_dist = float(np.linalg.norm(b_c - f_c))
    if contact_dist < 1e-9:
        return []
    if open_dir is None:
        _, _, vh = np.linalg.svd(pattern - p_c, full_matrices=False)
        pattern_normal = vh[-1]
    else:
        pattern_normal = np.asarray(open_dir, dtype=np.float32)
        n = float(np.linalg.norm(pattern_normal))
        if n < 1e-9:
            return []
        pattern_normal = pattern_normal / n
    signed_depth = float(p_c @ pattern_normal)
    contact_depth = abs(signed_depth)
    jaw_width = contact_dist + 2.0 * clearance - 2.0 * contact_depth
    if jaw_width < 0.0:
        return []
    tangent_offset = p_c - signed_depth * pattern_normal
    center = (f_c + b_c) * 0.5
    out = []
    for contact_pts, opposite_c in ((front, b_c), (back, f_c)):
        target_normal = opposite_c - contact_pts.mean(axis=0)
        target_normal = target_normal / (
            np.linalg.norm(target_normal) + 1e-9)
        aug_pat = np.vstack([pattern, [p_c + pattern_normal]])
        aug_wld = np.vstack([contact_pts, [contact_pts.mean(axis=0) + target_normal]])
        P = aug_pat - aug_pat.mean(axis=0)
        W = aug_wld - aug_wld.mean(axis=0)
        H = P.T @ W
        U, _, Vt = np.linalg.svd(H)
        R_pat = Vt.T @ U.T
        if np.linalg.det(R_pat) < 0:
            Vt[-1, :] *= -1
            R_pat = Vt.T @ U.T
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_pat.astype(np.float32)
        pose[:3, 3] = (center - pose[:3, :3] @ tangent_offset).astype(
            np.float32)
        out.append((pose, jaw_width))
    return out


def _build_collision_detector(gripper, target_sobj):
    """Build the gripper-vs-target collision batch (mirrors antipodal)
    so callers can prune colliding grasps in a single batch call."""
    items = gripper.runtime_lnks + [target_sobj]
    tgt_idx = len(items) - 1
    pairs = [(i, tgt_idx) for i in range(len(gripper.runtime_lnks))]
    try:
        detector = ocgcb.create_detector()
        batch = ocgcb.build_batch(items, pairs)
    except Exception:
        detlib = occs
        detector = detlib.create_detector()
        batch = detlib.build_batch(items, pairs)
    return detector, batch


def _check_grasp_collision(gripper, detector, batch, pose, jaw):
    """Place gripper at (pose, jaw) and return True if collided."""
    gripper.grip_at(
        pose[:3, 3], pose[:3, :3], float(jaw))
    return detector.detect_collision_batch(batch) is not None


# ---------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------

def sample_pattern(pattern, tgt_vs, tgt_fs, n_samples,
                   normal_tol_deg=0,
                   distance_tol=0.001,
                   surface_density_factor=1,
                   exclude_regions=None):
    """Sample N-point pattern placements on the target mesh surface.

    Every placement's contacts share a surface normal (within
    `normal_tol_deg`) and lie in that anchor's tangent plane (within
    `tgt_d * sin(normal_tol_deg)`). Pairwise distances must match the
    pattern's pairwise distances within `distance_tol`.

    pattern: (N, 3) contact pad positions in the TCP frame, coplanar.
    tgt_vs, tgt_fs: target mesh vertices/faces (object local frame).
    n_samples: max number of placements to return.
    exclude_regions: optional list of convex regions carved out of the
        candidate pool (see `osgop.clip_mesh`).

    Returns: list of N-tuples of float32 (3,) arrays in target frame.
    """
    pattern = np.asarray(pattern, dtype=np.float32)
    if pattern.ndim != 2 or pattern.shape[1] != 3:
        raise ValueError("pattern must be (N, 3)")
    n_contacts = pattern.shape[0]
    if n_contacts < 2:
        raise ValueError("pattern must have at least 2 points")
    pattern_pair_dists = np.linalg.norm(
        pattern[:, None, :] - pattern[None, :, :], axis=2
    ).astype(np.float32)

    pool_vs, pool_fs = osgop.clip_mesh(tgt_vs, tgt_fs, exclude_regions)
    if len(pool_fs) == 0:
        return []
    n_surf = max(surface_density_factor * n_samples, 5000)
    surf_pts, surf_nrms, _ = osgop.sample_surface(
        pool_vs, pool_fs, int(n_surf))
    surf_pts = np.asarray(surf_pts, dtype=np.float32)
    surf_nrms = np.asarray(surf_nrms, dtype=np.float32)
    surf_nrms = surf_nrms / (
        np.linalg.norm(surf_nrms, axis=1, keepdims=True) + 1e-9)
    if len(surf_pts) < n_contacts:
        return []

    cos_th = float(np.cos(np.deg2rad(normal_tol_deg)))
    sin_th = float(np.sin(np.deg2rad(normal_tol_deg)))
    kd = cKDTree(surf_pts)
    rng = np.random.default_rng()

    out = []
    for anchor_idx in rng.permutation(len(surf_pts)):
        anchor_pos = surf_pts[anchor_idx]
        anchor_n = surf_nrms[anchor_idx]
        placed_idx = [int(anchor_idx)]
        placed_pos = [anchor_pos]
        ok = True
        for k in range(1, n_contacts):
            tgt_d = float(pattern_pair_dists[0, k])
            inner = set(kd.query_ball_point(
                anchor_pos, tgt_d - distance_tol))
            outer = set(kd.query_ball_point(
                anchor_pos, tgt_d + distance_tol))
            cands = [j for j in (outer - inner) if j not in placed_idx]
            for i in range(1, k):
                di_tgt = float(pattern_pair_dists[i, k])
                fixed = placed_pos[i]
                cands = [
                    j for j in cands
                    if abs(np.linalg.norm(surf_pts[j] - fixed) - di_tgt)
                    <= distance_tol
                ]
                if not cands:
                    break
            if cands:
                cands = [
                    j for j in cands
                    if float(surf_nrms[j] @ anchor_n) >= cos_th
                ]
            if cands:
                proj_lim = tgt_d * sin_th
                cands = [
                    j for j in cands
                    if abs(float((surf_pts[j] - anchor_pos) @ anchor_n))
                    <= proj_lim
                ]
            if not cands:
                ok = False
                break
            j = int(rng.choice(cands))
            placed_idx.append(j)
            placed_pos.append(surf_pts[j])
        if ok:
            out.append(tuple(
                np.asarray(p, dtype=np.float32) for p in placed_pos))
            if len(out) >= n_samples:
                break
    return out


def pair_pattern(samples, tgt_vs, tgt_fs,
                 normal_tol_deg=0,
                 distance_tol=0.001,
                 min_thickness=0.0,
                 max_thickness=None):
    """For every front-side N-tuple in `samples`, ray-cast each contact
    along the inward surface normal to find a back-side hit. Keep the
    placements whose back-side hits also form a valid pattern (same
    pairwise distances + shared normal within tol). Returns a list of
    (front_tuple, back_tuple).

    min/max_thickness bounds the front-to-back ray distance.
    """
    if not samples:
        return []
    tgt_vs = np.asarray(tgt_vs, dtype=np.float32)
    tgt_fs = np.asarray(tgt_fs, dtype=np.int32)
    v0 = tgt_vs[tgt_fs[:, 0]]
    v1 = tgt_vs[tgt_fs[:, 1]]
    v2 = tgt_vs[tgt_fs[:, 2]]
    fns = np.cross(v1 - v0, v2 - v0)
    fns = fns / (np.linalg.norm(fns, axis=1, keepdims=True) + 1e-9)
    centers = (v0 + v1 + v2) / 3.0
    kd = cKDTree(centers)
    cos_th = float(np.cos(np.deg2rad(normal_tol_deg)))
    sin_th = float(np.sin(np.deg2rad(normal_tol_deg)))
    eps_offset = 1e-5

    out = []
    for sample in samples:
        front_pts = np.asarray(sample, dtype=np.float32)
        # Anchor outward normal: nearest face's normal at front[0].
        _, anchor_face = kd.query(front_pts[0], k=1)
        anchor_n = fns[anchor_face].astype(np.float32)
        ray_dir = -anchor_n
        origins = front_pts + eps_offset * ray_dir
        directions = np.broadcast_to(
            ray_dir, front_pts.shape).copy()
        hit_t, hit_id = _ray_triangles_batch_far(
            origins, directions, v0, v1, v2)
        if np.any(hit_id < 0):
            continue
        if max_thickness is not None and np.any(hit_t > max_thickness):
            continue
        if np.any(hit_t < min_thickness):
            continue
        back_pts = origins + hit_t[:, None] * directions
        back_nrms = fns[hit_id]
        if np.any((back_nrms @ ray_dir) < cos_th):
            continue
        front_d = np.linalg.norm(
            front_pts[:, None, :] - front_pts[None, :, :], axis=2)
        back_d = np.linalg.norm(
            back_pts[:, None, :] - back_pts[None, :, :], axis=2)
        if np.any(np.abs(front_d - back_d) > distance_tol):
            continue
        # Back contacts (approximately) coplanar wrt the back anchor.
        back_anchor_n = back_nrms[0]
        ok = True
        for k in range(1, len(back_pts)):
            d = float(np.linalg.norm(back_pts[k] - back_pts[0]))
            if d == 0.0:
                continue
            if abs(float(
                    (back_pts[k] - back_pts[0]) @ back_anchor_n)
                   ) > d * sin_th:
                ok = False
                break
        if not ok:
            continue
        out.append((tuple(front_pts), tuple(back_pts)))
    return out


def polypodal(gripper, target_sobj, n_samples,
              normal_tol_deg=0,
              distance_tol=0.001,
              surface_density_factor=1,
              exclude_regions=None,
              clearance=0.0003,
              min_thickness=0.0,
              max_thickness=None,
              verbose=True,
              return_pairs=False):
    """End-to-end polypodal grasp computation.

        sample_pattern -> pair_pattern -> _hand_poses_from_pair
        -> reject gripper-vs-target collisions.

    Returns: list of (pose_4x4, jaw_width). If return_pairs is True,
        returns (pose_4x4, jaw_width, front_tuple, back_tuple).
    """
    pattern = getattr(gripper, 'contact_pattern', None)
    if pattern is None:
        raise ValueError('gripper must define contact_pattern')
    pattern = np.asarray(pattern, dtype=np.float32)
    if pattern.ndim != 2 or pattern.shape[1] != 3 or pattern.shape[0] < 2:
        raise ValueError('polypodal requires gripper.contact_pattern to be (N, 3), N >= 2')
    tgt_vs, tgt_fs, _ = occs.cols_to_vffns(target_sobj.collisions)
    target_tf = target_sobj.wd_tf
    open_dir = getattr(gripper, 'open_dir', None)
    samples = sample_pattern(
        pattern, tgt_vs, tgt_fs, n_samples,
        normal_tol_deg=normal_tol_deg,
        distance_tol=distance_tol,
        surface_density_factor=surface_density_factor,
        exclude_regions=exclude_regions)
    pairs = pair_pattern(
        samples, tgt_vs, tgt_fs,
        normal_tol_deg=normal_tol_deg,
        distance_tol=distance_tol,
        min_thickness=min_thickness,
        max_thickness=max_thickness)
    gripper = gripper.clone()
    detector, batch = _build_collision_detector(gripper, target_sobj)
    out = []
    for front, back in pairs:
        for pose, jaw in _hand_poses_from_pair(
                pattern, front, back, open_dir, clearance):
            jaw_lo, jaw_hi = gripper.jaw_range
            if jaw < jaw_lo or jaw > jaw_hi:
                continue
            pose = (target_tf @ pose).astype(np.float32)
            if _check_grasp_collision(gripper, detector, batch, pose, jaw):
                continue
            if verbose:
                front_arr = np.asarray(front, dtype=np.float32)
                back_arr = np.asarray(back, dtype=np.float32)
                pair_dist = float(np.linalg.norm(
                    back_arr.mean(axis=0) - front_arr.mean(axis=0)))
                print(
                    f"polypodal grasp {len(out) + 1}: "
                    f"opening_width={jaw * 1000.0:.3f}mm, "
                    f"pair_dist={pair_dist * 1000.0:.3f}mm")
            if return_pairs:
                out.append((pose, jaw, front, back))
            else:
                out.append((pose, jaw))
    return out


# ---------------------------------------------------------------------
# Demo: visualize the pipeline on con_ma. SPACE cycles candidates.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import builtins
    import pyglet.window.key as pkey

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import one.scene.scene_object as osso
    import one.scene.scene_object_primitive as ossop
    import one.utils.constant as ouc
    import one.viewer.world as ovw
    from kurabo.grippers.krb_right.krb_right import KRBRight

    OBJ = os.path.join(_root, "kurabo", "objects", "con_ma.stl")
    N_SAMPLES = 5000
    CLEARANCE = 0.0003
    SPHERE_R = 0.0004
    EXCLUDE_REGIONS = [
        [((0.0, 0.001, 0.0), (0.0,  1.0, 0.0))],   # cut: y < 1mm
        [((0.0, 0.009, 0.0), (0.0, -1.0, 0.0))],   # cut: y > 9mm
    ]

    obj = osso.SceneObject.from_file(
        OBJ, collision_type=ouc.CollisionType.MESH, is_free=True)
    obj.pos = np.zeros(3, dtype=np.float32)
    obj.rotmat = np.eye(3, dtype=np.float32)
    ghost = KRBRight()

    poses = polypodal(
        ghost, obj, N_SAMPLES,
        normal_tol_deg=15,
        distance_tol=0.001,
        surface_density_factor=1,
        exclude_regions=EXCLUDE_REGIONS,
        clearance=CLEARANCE,
        min_thickness=0.001,
        max_thickness=0.05,
        return_pairs=True)
    print(f"grasps={len(poses)} (collision-free)")
    if not poses:
        sys.exit(0)
    n_tup = len(poses)
    n_pts_per = len(poses[0][2])

    base = ovw.World(cam_pos=(0.05, 0.05, 0.05),
                     cam_lookat_pos=(0.0, 0.0, 0.0))
    builtins.base = base
    obj.rgb = (0.85, 0.6, 0.3)
    obj.alpha = 0.5
    obj.attach_to(base.scene)
    front_spheres = [
        ossop.sphere(pos=poses[0][2][k], radius=SPHERE_R,
                     rgb=ouc.BasicColor.MAGENTA)
        for k in range(n_pts_per)]
    back_spheres = [
        ossop.sphere(pos=poses[0][3][k], radius=SPHERE_R,
                     rgb=ouc.BasicColor.CYAN)
        for k in range(n_pts_per)]
    for s in front_spheres + back_spheres:
        s.attach_to(base.scene)
    ghost.attach_to(base.scene)
    jaw_lo, jaw_hi = ghost.jaw_range
    cursor = [0]

    def _show(i):
        pose, jaw, front_pts, back_pts = poses[i]
        for k in range(n_pts_per):
            front_spheres[k].pos = front_pts[k]
            back_spheres[k].pos = back_pts[k]
        ghost.grip_at(
            tgt_pos=pose[:3, 3], tgt_rotmat=pose[:3, :3],
            tgt_jaw_width=float(np.clip(jaw, jaw_lo, jaw_hi)))
        msg = f"pair {i + 1} / {n_tup}  jaw={jaw * 1000:.1f}mm"
        base.set_caption(msg)
        print(msg)

    _show(0)

    REPEAT_DELAY = 0.4
    held_for = [0.0]

    def _step(dt):
        im = base.input_manager
        if im.is_key_pressed_edge(pkey.SPACE):
            cursor[0] = (cursor[0] + 1) % n_tup
            _show(cursor[0])
            held_for[0] = 0.0
        elif im.is_key_pressed(pkey.SPACE):
            held_for[0] += dt
            if held_for[0] >= REPEAT_DELAY:
                cursor[0] = (cursor[0] + 1) % n_tup
                _show(cursor[0])
        else:
            held_for[0] = 0.0

    base.schedule_interval(_step, interval=0.05)
    base.run()
