"""Monocontact: single-contact ("one contact pad") surface-approach
planning for suction / tip tools.

Naming convention in this package -- the suffix encodes the *mechanism*,
the prefix the *contact count*:

    -podal   : opposing pinch, force-closure (the pads press against
               each other).  antipodal (2), polypodal (N).
    -contact : same-side adhesion / press, NOT force-closure (suction,
               magnetic, a tip pressing or inserting).  monocontact (1),
               and a future polycontact (N, e.g. a suction-cup array).

A monocontact grasp therefore has a single contact and no opposition --
the tool is held against the surface by suction / adhesion, or simply
presses on / inserts into it. The planner samples the target surface
and, for each sampled point, aligns the tool's contact axis (the tcp
local +z) with the inward surface normal, optionally rolls about that
axis, then rejects tool-vs-target collisions. It needs only a tcp to
align and ``runtime_lnks`` for collision, so it works for any
single-contact end effector (suction cup, screwdriver tip, probe, ...).

Outputs are (pose_4x4, pre_pose_4x4, score) tuples; ``pose`` places the
tcp origin at the contact point with +z pointing into the surface, and
``pre_pose`` is the same pose retreated along the approach axis.
"""
import numpy as np

import one.utils.math as oum
import one.scene.geometry_ops as osgop
import one.collider.cpu_simd as occs
import one.grasp._common as ogc


def monocontact_iter(tool, target_sobj, tcp='tip',
                     density=0.02, roll_step_deg=90, retreat=None,
                     approach_bias=(0.0, 0.0, 1.0), exclude_regions=None):
    """
    Generator: yields (pose_tf, pre_pose_tf, score, collided).
    :param tool: single-contact end effector (must expose ``tcp(name)``,
        ``runtime_lnks`` and ``set_pos_rotmat``)
    :param target_sobj: target object to contact
    :param tcp: name of the contact tcp to align (default 'tip')
    :param density: surface sampling density (smaller -> denser)
    :param roll_step_deg: roll step about the approach axis, in degrees.
        For an axisymmetric suction cup one roll is enough; finer steps
        only matter for an asymmetric tool body's collisions.
    :param retreat: pre-pose retreat distance along the approach axis.
        Defaults to half the tcp offset length.
    :param approach_bias: world direction favoured by the score; the
        default world +z rewards top-facing surfaces (a suction seal
        approached from above). Set to None to score all contacts equally.
    :param exclude_regions: optional convex regions carved out of the
        contact-sampling surface (full mesh still used for collisions).
    :return: yields (pose_tf, pre_pose_tf, score, collided). ``pose``
        aligns tcp +z into the surface at the contact point.
    """
    tool = tool.clone()
    tcp_loc = np.asarray(tool.tcp(tcp).loc_tf, dtype=np.float32)
    if retreat is None:
        retreat = 0.5 * float(np.linalg.norm(tcp_loc[:3, 3]))
    tcp_loc_inv = np.linalg.inv(tcp_loc)

    # Plan in the target's LOCAL (zero-pose) frame: clone the target and zero
    # its pose so contact sampling (local geom) and the tool-vs-target collision
    # check (CollisionBatch uses target.tf) share one frame. Returned poses are
    # in the target's local frame; the caller maps them onto the placed object.
    target_sobj = target_sobj.clone()
    target_sobj.set_pos_rotmat(
        pos=np.zeros(3, dtype=np.float32), rotmat=np.eye(3, dtype=np.float32))
    tgt_vs, tgt_fs, _ = occs.cols_to_vffns(target_sobj.collisions)
    if exclude_regions:
        tgt_vs, tgt_fs = osgop.clip_mesh(tgt_vs, tgt_fs, exclude_regions)
        if len(tgt_fs) == 0:
            return
    n_samples = osgop.sample_count_from_area(tgt_vs, tgt_fs, density)
    pts, nrms, _ = osgop.sample_surface(tgt_vs, tgt_fs, n_samples)
    nrms = nrms / (np.linalg.norm(nrms, axis=1, keepdims=True) + oum.eps)

    # approach (tool +z) points into the surface, opposite the outward normal
    approach = -nrms
    rot_base = oum.frame_from_normal(approach)                  # (N,3,3), z=approach
    roll_step = np.deg2rad(roll_step_deg)
    angles = np.arange(0.0, 2.0 * np.pi, roll_step)
    roll_rots = oum.rotmat_from_axangle(
        approach[:, None, :], angles[None, :])                 # (N,K,3,3)
    rot_all = roll_rots @ rot_base[:, None, :, :]               # (N,K,3,3)

    pose_tf = np.tile(np.eye(4, dtype=np.float32),
                      (rot_all.shape[0], rot_all.shape[1], 1, 1))
    pose_tf[:, :, :3, :3] = rot_all
    pose_tf[:, :, :3, 3] = pts[:, None, :]
    pose_all = pose_tf.reshape(-1, 4, 4)

    if approach_bias is None:
        score = np.zeros(len(nrms), dtype=np.float32)
    else:
        bias = np.asarray(approach_bias, dtype=np.float32)
        bias = bias / (np.linalg.norm(bias) + oum.eps)
        score = 0.5 * (1.0 + nrms @ bias)        # top-facing -> 1, down -> 0
    score_all = np.repeat(score, len(angles))
    order = np.argsort(score_all)[::-1]
    pose_all = pose_all[order]
    score_all = score_all[order]

    # tool-vs-target collision batch
    detector, batch = ogc.build_ee_target_detector(tool, target_sobj)

    for pose, sc in zip(pose_all, score_all):
        collided = False
        # contact pose
        base_tf = pose @ tcp_loc_inv
        tool.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        if detector.detect_collision_batch(batch) is not None:
            collided = True
        # pre-contact (retreated) pose: move back along +outward normal
        pre_pose = pose.copy()
        pre_pose[:3, 3] = pose[:3, 3] - retreat * pose[:3, 2]
        pre_base = pre_pose @ tcp_loc_inv
        tool.set_pos_rotmat(pre_base[:3, 3], pre_base[:3, :3])
        if detector.detect_collision_batch(batch) is not None:
            collided = True
        yield pose, pre_pose, float(sc), collided


def monocontact(tool, target_sobj, tcp='tip',
                density=0.02, roll_step_deg=90, retreat=None,
                max_grasps=50, approach_bias=(0.0, 0.0, 1.0),
                exclude_regions=None):
    """
    Collects non-colliding single-contact grasps only.
    :param tool: single-contact end effector
    :param target_sobj: target object to contact
    :param tcp: name of the contact tcp to align (default 'tip')
    :param density: surface sampling density
    :param roll_step_deg: roll step about the approach axis, in degrees
    :param retreat: pre-pose retreat distance (defaults to half tcp length)
    :param max_grasps: maximum number of grasps to return
    :param approach_bias: world direction favoured by the score
    :param exclude_regions: convex regions carved out of contact sampling
    :return: list of (pose_tf, pre_pose_tf, score)
    """
    results = []
    for pose, pre_pose, sc, collided in monocontact_iter(
            tool, target_sobj, tcp, density, roll_step_deg,
            retreat, approach_bias, exclude_regions=exclude_regions):
        if not collided:
            results.append((pose, pre_pose, float(sc)))
        if max_grasps is not None and len(results) >= max_grasps:
            break
    return results
