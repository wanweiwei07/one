"""Pick-and-place motion planning -- move an object from a pick pose to a place
pose, composed from the existing building blocks:

    GraspReasoner.reason_common_gids  -- a grasp reachable + collision-free at
                                         BOTH the pick and the place pose.
    ADPlanner.gen_approach / gen_depart -- RRT to a pre-grasp then a straight
                                         cartesian line into the grasp, mirrored.
    MotionData                         -- the composed trajectory (joint configs,
                                         gripper openings, AND the carried-object
                                         world pose at every waypoint).

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
import one.motion.primitives.approach_depart as ompad
import one.grasp.reasoner as ogr
from one.motion.core.motion_data import MotionData


def gen_pick_place(robot, gripper, obj, grasps, pick_pose, place_pose, *,
                   statics=(), tcp=None, chain='main', start_qs=None,
                   lift_height=0.12, granularity=0.01, margin=0.0,
                   approach_iters=4000, transfer_iters=8000, goal_bias=0.3,
                   jaw_to_qs=lambda w: (w / 2.0, w / 2.0)):
    """Plan home -> pick -> lift -> transfer -> place -> retreat moving ``obj``
    from ``pick_pose`` to ``place_pose`` (both (4,4) object world tfs).

    ``grasps``: object-LOCAL grasps (pose, pre_pose, jaw_width, score) as from
    ``antipodal`` / ``load_grasps``. ``statics``: the fixed scenery/obstacles
    (ground, walls) -- obstacles in BOTH phases. Tries each grasp feasible at
    both poses until one fully plans.

    ``tcp`` is NOT fixed by the caller: by default (None) the IK tcp is DERIVED
    from each grasp via ``gripper.eval_grasp_tcp(jaw_width)`` -- the grasp's own
    grasp-center frame (a DexHand's shifts with the closure; a parallel
    gripper's is fixed). Pass a name/TCP only to force one fixed tcp.

    Returns a MotionData with parallel ``jv_list`` (configs), ``ev_list`` (gripper
    opening per waypoint; open->closed marks the grasp, closed->open the release)
    and ``obj_pose_list`` (the object's world pose per waypoint: at pick_pose
    before the grasp, carried with the gripper, at place_pose after release), or
    None if no feasible grasp yields a full plan.
    """
    if isinstance(tcp, str):
        tcp = gripper.tcp(tcp)
    statics = list(statics)
    if start_qs is None:
        start_qs = np.asarray(robot.qs, dtype=np.float64).copy()
    pick_pose = np.asarray(pick_pose, dtype=np.float32)
    place_pose = np.asarray(place_pose, dtype=np.float32)
    jaw_open = float(gripper.jaw_range[1])

    # the grasp-center tcp for a grasp: derived from its jaw_width by default
    # (the gripper is the authority), or the caller's forced tcp.
    def grasp_tcp(jaw_width):
        return gripper.eval_grasp_tcp(jaw_width) if tcp is None else tcp

    # --- FREE collider: robot + gripper + statics (NOT the object) ---
    mjc_free = ocm.MJCollider()
    for e in (robot, gripper, *statics):
        mjc_free.append(e)
    mjc_free.actors = [robot]
    mjc_free.compile(margin=margin, auto_acm=True)
    base_acm = list(mjc_free.acm)
    ctx_free = omppc.PlanningContext(collider=mjc_free)
    planner_free = ompr.RRTConnectPlanner(pln_ctx=ctx_free, goal_bias=goal_bias)

    # --- grasps feasible at BOTH the pick and the place pose (the reasoner
    # derives each grasp's tcp the same way, from gripper.eval_grasp_tcp) ---
    reasoner = ogr.GraspReasoner(robot, ctx_free, grasps, tcp=tcp,
                                 gripper=gripper, chain=chain,
                                 jaw_to_qs=jaw_to_qs)
    common = reasoner.reason_common_gids([pick_pose, place_pose])

    for gid in sorted(common):
        pose, pre_pose, jw = grasps[gid][0], grasps[gid][1], grasps[gid][2]
        pose = np.asarray(pose, dtype=np.float32)
        g_tcp = grasp_tcp(jw)                    # this grasp's grasp-center frame
        pick_g, pick_pre = pick_pose @ pose, pick_pose @ np.asarray(pre_pose,
                                                                    np.float32)
        place_g, place_pre = place_pose @ pose, place_pose @ np.asarray(
            pre_pose, np.float32)

        # PICK (free phase, gripper open). Descent into the target is ungated.
        adp_free = ompad.ADPlanner(robot, ctx_free, planner_free,
                                   chain=chain, tcp=g_tcp)
        mjc_free.set_mecba_qpos(gripper, jaw_to_qs(jaw_open))
        ctx_free.clear_cache()
        pick = adp_free.gen_approach(
            pick_g[:3, 3], pick_g[:3, :3], start_qs=start_qs,
            pre_pos=pick_pre[:3, 3], pre_rotmat=pick_pre[:3, :3],
            granularity=granularity, use_rrt=True, check_descent=False,
            ee_value=jaw_open, max_iters=approach_iters)
        if pick is None:
            continue
        q_grasp = pick.jv_list[-1]

        # GRASP: mount the object so it rides the gripper, build the CARRY
        # collider (object collision-checked), rule-based ACM.
        obj.pos, obj.rotmat = pick_pose[:3, 3], pick_pose[:3, :3]
        robot.fk(qs=q_grasp)
        gripper.grasp(obj, jaw_width=jw)
        mjc_carry = ocm.MJCollider()
        for e in (robot, *statics):
            mjc_carry.append(e)
        mjc_carry.actors = [robot]
        mjc_carry.compile(margin=margin, auto_acm=False,
                          extra_excludes=base_acm
                          + [(obj, lnk) for lnk in gripper.runtime_lnks])
        ctx_carry = omppc.PlanningContext(collider=mjc_carry)
        planner_carry = ompr.RRTConnectPlanner(pln_ctx=ctx_carry,
                                               goal_bias=goal_bias)
        adp_carry = ompad.ADPlanner(robot, ctx_carry, planner_carry,
                                    chain=chain, tcp=g_tcp)
        mjc_carry.set_mecba_qpos(gripper, jaw_to_qs(jw))
        ctx_carry.clear_cache()

        # LIFT (carry), TRANSFER+PLACE (gated RRT -> held obj avoids obstacles),
        # RETREAT (released). Cartesian legs ungated (intended pick/place contact).
        lift = adp_carry.gen_depart(
            pick_g[:3, 3], pick_g[:3, :3], start_qs=q_grasp,
            depart_direction=(0, 0, 1), depart_distance=lift_height,
            granularity=granularity, ee_value=jw, check_retreat=False)
        place = retreat = None
        if lift is not None:
            place = adp_carry.gen_approach(
                place_g[:3, 3], place_g[:3, :3], start_qs=lift.jv_list[-1],
                pre_pos=place_pre[:3, 3], pre_rotmat=place_pre[:3, :3],
                granularity=granularity, use_rrt=True, check_descent=False,
                ee_value=jw, max_iters=transfer_iters)
        if place is not None:
            retreat = adp_carry.gen_depart(
                place_g[:3, 3], place_g[:3, :3], start_qs=place.jv_list[-1],
                depart_direction=(0, 0, 1), depart_distance=lift_height,
                granularity=granularity, ee_value=jaw_open, check_retreat=False)
        gripper.release(obj)                    # drop the planning mount
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
        for i, q in enumerate(motion.jv_list):
            if i < grasp_idx:
                tf = pick_pose.astype(np.float64)
            elif i <= release_idx:
                robot.fk(qs=q)
                tf = np.asarray(g_tcp.tf, dtype=np.float64) @ obj_in_grasp
            else:
                tf = place_pose.astype(np.float64)
            obj_poses.append(tf.astype(np.float32))
        return MotionData(motion.jv_list, motion.ev_list, obj_poses)
    return None


class PickPlacePlanner:
    """Thin stateful facade over :func:`gen_pick_place`.

    Binds the cell -- ``robot``, ``gripper``, the static ``statics`` (ground,
    walls), the ``tcp`` / ``chain`` and the planning knobs -- once, so call sites
    pass only what varies per task: the object, its grasps, and the pick/place
    poses. The free function stays the testable core; this is pure ergonomics
    (cf. ADPlanner / GraspReasoner). Any bound knob is overridable per call.
    """

    def __init__(self, robot, gripper, *, statics=(), tcp=None,
                 chain='main', lift_height=0.12, granularity=0.01, margin=0.0,
                 approach_iters=4000, transfer_iters=8000, goal_bias=0.3,
                 jaw_to_qs=lambda w: (w / 2.0, w / 2.0)):
        self.robot = robot
        self.gripper = gripper
        self._defaults = dict(
            statics=tuple(statics), tcp=tcp, chain=chain,
            lift_height=lift_height, granularity=granularity, margin=margin,
            approach_iters=approach_iters, transfer_iters=transfer_iters,
            goal_bias=goal_bias, jaw_to_qs=jaw_to_qs)

    def gen_pick_place(self, obj, grasps, pick_pose, place_pose, **overrides):
        kw = dict(self._defaults)
        kw.update(overrides)
        return gen_pick_place(self.robot, self.gripper, obj, grasps,
                              pick_pose, place_pose, **kw)
