"""Pick-and-place motion planning -- move an object from a pick pose to a place
pose, composed from the existing building blocks:

    GraspReasoner.reason_common_gids  -- a grasp reachable + collision-free at
                                         BOTH the pick and the place pose.
    gen_approach / gen_depart          -- RRT to a pre-grasp then a straight
                                         cartesian line into the grasp, mirrored.
    MotionData                         -- the composed trajectory (robot configs,
                                         the EE joint config, AND the carried
                                         object's world pose at every waypoint).

Usually invoked through ``robot.pick_place(...)`` (the arm IS the manipulator;
the gripper is its ``end_effector``); this free function is the testable core.

The crux is TWO-PHASE collision:

  FREE phase (reach to pick): the object is the manipulated target, NOT an
    obstacle (the grasp set already guarantees its clearance), so the planning
    collider is robot + gripper + statics. auto_acm detects the arm's structural
    self-collisions once; that ACM is reused by the carry collider.

  CARRY phase (lift / transfer / place): the object is mounted on the gripper and
    IS collision-checked -- it rides the gripper link in the model, so the
    transfer routes the HELD object (not just the gripper) around obstacles. The
    carry ACM is rule-based: reuse the base ACM + an explicit (object <-> gripper
    links) exemption (the intended grasp contact), NO re-detection.

The cartesian descent into the grasp and the cartesian lift/place legs are
ungated (the object intentionally contacts the pick/place target there, like
l1picking); obstacle avoidance happens on the gated RRT travel between them.
``margin`` inflates obstacles in both colliders (guard against thin obstacles /
discrete-sampling tunnelling).
"""
import numpy as np

import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.motion.primitives.approach_depart import gen_approach, gen_depart
import one.grasp.reasoner as ogr
from one.motion.core.motion_data import MotionData


def gen_pick_place(robot, gripper, obj, grasps, pick_pose, place_pose, *,
                   statics=(), tcp=None, chain='main', start_qs=None,
                   lift_height=0.12, granularity=0.01, margin=0.0,
                   approach_iters=4000, transfer_iters=8000, goal_bias=0.3):
    """Plan home -> pick -> lift -> transfer -> place -> retreat moving ``obj``
    from ``pick_pose`` to ``place_pose`` (both (4,4) object world tfs).

    ``grasps``: object-LOCAL :class:`~one.grasp.grasp.Grasp` records (as from
    ``antipodal`` / ``load_grasps``). ``statics``: the fixed scenery/obstacles
    (ground, walls) -- obstacles in BOTH phases. Tries each grasp feasible at
    both poses until one fully plans.

    ``tcp`` is NOT fixed by the caller: by default (None) the IK tcp is each
    grasp's OWN frozen grasp-center frame (``grasp.make_tcp(gripper)``) -- no
    re-derivation from the gripper's current mode / closure, so a dexterous
    hand's per-grasp tcp is honored unambiguously. Pass a name/TCP only to force
    one fixed tcp. The gripper is posed from each grasp's frozen ``qpos`` /
    ``pre_qpos`` for collision (no width->qs lambda needed).

    Returns a MotionData with parallel ``robot_qpos_list`` (arm configs),
    ``ee_qpos_list`` (the end-effector joint config per waypoint; open->closed
    marks the grasp, closed->open the release) and ``obj_pose_list`` (the
    object's world pose per waypoint: at pick_pose before the grasp, carried with
    the gripper, at place_pose after release), or None if no feasible grasp
    yields a full plan.
    """
    if isinstance(tcp, str):
        tcp = gripper.tcp(tcp)
    # Grasps are END-EFFECTOR-bound: their frozen tcp / qpos were planned for a
    # specific EE (e.g. ``hand.as_jaw('pinch')``). Reject a mismatched gripper
    # LOUDLY -- otherwise an incompatible frozen tcp/qpos would be applied to the
    # wrong root and silently produce garbage motion.
    ee_name = type(gripper).__name__
    mismatched = sorted({g.provenance.get("ee") for g in grasps
                         if g.provenance.get("ee") not in (None, ee_name)})
    if mismatched:
        raise ValueError(
            f"grasps were planned for {mismatched}, but the gripper is "
            f"{ee_name!r}; plan grasps with THIS end effector (a parallel "
            f"gripper directly, or ``gripper.as_jaw(<mode>)`` for a hand).")
    statics = list(statics)
    if start_qs is None:
        start_qs = np.asarray(robot.qs, dtype=np.float64).copy()
    pick_pose = np.asarray(pick_pose, dtype=np.float32)
    place_pose = np.asarray(place_pose, dtype=np.float32)

    # Plan ONLY the ``chain``: freeze every robot joint NOT on it at its
    # ``start_qs`` value. A no-op for an arm whose chain spans all its joints;
    # essential for a humanoid (keeps the torso / other arm still while the
    # planning arm reaches), so the high-level planner matches a hand-rolled
    # chain-restricted context.
    joint_limits = robot.chain_joint_limits(chain, start_qs)

    # --- FREE collider: robot + gripper + statics (NOT the object) ---
    mjc_free = ocm.MJCollider()
    for e in (robot, gripper, *statics):
        mjc_free.append(e)
    mjc_free.actors = [robot]
    mjc_free.compile(margin=margin, auto_acm=True)
    base_acm = list(mjc_free.acm)
    ctx_free = omppc.PlanningContext(collider=mjc_free, joint_limits=joint_limits)
    planner_free = ompr.RRTConnectPlanner(pln_ctx=ctx_free, goal_bias=goal_bias)

    # --- grasps feasible at BOTH the pick and the place pose (the reasoner
    # uses each grasp's OWN frozen tcp, no re-derivation) ---
    reasoner = ogr.GraspReasoner(robot, ctx_free, grasps, tcp=tcp,
                                 gripper=gripper, chain=chain)
    common = reasoner.reason_common_gids([pick_pose, place_pose])

    # end-effector OPEN config for the free reach phase -- uniform across a
    # parallel gripper and a dexterous hand via the shared ``open``.
    gripper.open()
    open_qpos = np.asarray(gripper.qs, dtype=np.float32).copy()

    for gid in sorted(common):
        grasp = grasps[gid]
        pose, pre_pose = grasp.pose, grasp.pre_pose
        g_tcp = grasp.make_tcp(gripper) if tcp is None else tcp
        pick_g, pick_pre = pick_pose @ pose, pick_pose @ pre_pose
        place_g, place_pre = place_pose @ pose, place_pose @ pre_pose

        # PICK (free phase, gripper open). Descent into the target is ungated.
        mjc_free.set_mecba_qpos(gripper, open_qpos)
        ctx_free.clear_cache()
        pick = gen_approach(
            robot, ctx_free, planner_free, pick_g[:3, 3], pick_g[:3, :3],
            tcp=g_tcp, start_qs=start_qs, chain=chain,
            pre_pos=pick_pre[:3, 3], pre_rotmat=pick_pre[:3, :3],
            granularity=granularity, use_rrt=True, check_descent=False,
            ee_qpos=open_qpos, max_iters=approach_iters)
        if pick is None:
            continue
        q_grasp = pick.robot_qpos_list[-1]

        # GRASP: mount the object so it rides the gripper, build the CARRY
        # collider (object collision-checked), rule-based ACM.
        obj.pos, obj.rotmat = pick_pose[:3, 3], pick_pose[:3, :3]
        robot.fk(qs=q_grasp)
        gripper.attach(obj, grasp.qpos)          # frozen closure, no jaw_width
        mjc_carry = ocm.MJCollider()
        for e in (robot, *statics):
            mjc_carry.append(e)
        mjc_carry.actors = [robot]
        mjc_carry.compile(margin=margin, auto_acm=False,
                          extra_excludes=base_acm
                          + [(obj, lnk) for lnk in gripper.runtime_lnks])
        ctx_carry = omppc.PlanningContext(collider=mjc_carry,
                                          joint_limits=joint_limits)
        planner_carry = ompr.RRTConnectPlanner(pln_ctx=ctx_carry,
                                               goal_bias=goal_bias)
        mjc_carry.set_mecba_qpos(gripper, grasp.qpos)
        ctx_carry.clear_cache()

        # LIFT (carry), TRANSFER+PLACE (gated RRT -> held obj avoids obstacles),
        # RETREAT (released). Cartesian legs ungated (intended pick/place contact).
        lift = gen_depart(
            robot, ctx_carry, planner_carry, pick_g[:3, 3], pick_g[:3, :3],
            tcp=g_tcp, start_qs=q_grasp, chain=chain,
            depart_direction=(0, 0, 1), depart_distance=lift_height,
            granularity=granularity, ee_qpos=grasp.qpos, check_retreat=False)
        place = retreat = None
        if lift is not None:
            place = gen_approach(
                robot, ctx_carry, planner_carry, place_g[:3, 3], place_g[:3, :3],
                tcp=g_tcp, start_qs=lift.robot_qpos_list[-1], chain=chain,
                pre_pos=place_pre[:3, 3], pre_rotmat=place_pre[:3, :3],
                granularity=granularity, use_rrt=True, check_descent=False,
                ee_qpos=grasp.qpos, max_iters=transfer_iters)
        if place is not None:
            retreat = gen_depart(
                robot, ctx_carry, planner_carry, place_g[:3, 3], place_g[:3, :3],
                tcp=g_tcp, start_qs=place.robot_qpos_list[-1], chain=chain,
                depart_direction=(0, 0, 1), depart_distance=lift_height,
                granularity=granularity, ee_qpos=open_qpos, check_retreat=False)
        gripper.detach(obj)                     # drop the planning mount
        if retreat is None:
            continue

        before_release = pick + lift + place
        grasp_idx = len(pick) - 1               # gripper closes here
        release_idx = len(before_release) - 1   # gripper opens here
        motion = before_release + retreat

        # carried-object world pose per waypoint: pick_pose until the grasp, then
        # rigid on the grasp-center frame, then place_pose after release. The
        # object's pose in the grasp frame is just inv(grasp local pose).
        obj_in_grasp = np.linalg.inv(pose.astype(np.float64))
        obj_poses = []
        for i, q in enumerate(motion.robot_qpos_list):
            if i < grasp_idx:
                tf = pick_pose.astype(np.float64)
            elif i <= release_idx:
                robot.fk(qs=q)
                tf = np.asarray(g_tcp.tf, dtype=np.float64) @ obj_in_grasp
            else:
                tf = place_pose.astype(np.float64)
            obj_poses.append(tf.astype(np.float32))
        return MotionData(motion.robot_qpos_list, motion.ee_qpos_list, obj_poses)
    return None
