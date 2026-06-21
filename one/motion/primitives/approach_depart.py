"""Approach / depart motion primitives.

The single pattern every pick, place and insertion reuses:

    PROBABILISTIC plan (RRT) to a *pre-grasp* pose, then a straight CARTESIAN
    line into the grasp -- so the free-space travel is collision-planned but the
    final centimetres come in straight along the approach axis (the hand does not
    bow sideways into a wall the way a joint-space line would).

``gen_approach`` builds that (RRT/joint travel + cartesian descent); ``gen_depart``
is its mirror (cartesian retreat + optional RRT to a parking config). Both return
a :class:`~one.motion.core.motion_data.MotionData` so segments compose with ``+``.

These wrap the existing primitives -- ``ompr.RRTConnectPlanner.solve`` (or
``omij.linear_path``) for the travel and ``omic.linear_to_jpath`` for the
cartesian leg -- and the standard ``PlanningContext`` collision gate; they add no
new collision/IK machinery, just the sequencing that examples used to inline.

Conventions: pos-first; ``tcp`` is the registered tcp name or a ``TCP`` object IK
solves against; configs are full qs (the planners' state vector). The approach
axis defaults to the grasp frame's +z (``goal_rotmat[:, 2]``).
"""
import numpy as np

import one.motion.interpolation.cartesian as omic
import one.motion.interpolation.joint as omij
from one.motion.core.motion_data import MotionData


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def nearest_valid_ik(robot, ctx, pos, rotmat, *, chain='main', tcp='flange',
                     ref_qs, max_solutions=8, accept=None):
    """IK at ``(pos, rotmat)`` returning the solution nearest ``ref_qs`` (in the
    chain's active-joint space) that is collision-free under ``ctx`` (and passes
    the optional ``accept(full_qs)`` predicate). Full qs, or None if none qualify.
    ``ctx=None`` skips the collision gate (pure kinematics)."""
    ik_chain = robot.chain(chain)
    ref_active = ik_chain.extract_active_qs(np.asarray(ref_qs, dtype=np.float32))
    best, best_d = None, None
    for s in robot.ik(pos, rotmat, chain=chain, tcp=tcp,
                      ref_qs=ref_active, max_solutions=max_solutions):
        s64 = np.asarray(s, dtype=np.float64)
        if ctx is not None and not ctx.is_state_valid(s64):
            continue
        if accept is not None and not accept(s64):
            continue
        d = float(np.linalg.norm(ik_chain.extract_active_qs(s) - ref_active))
        if best_d is None or d < best_d:
            best, best_d = np.asarray(s, dtype=np.float32), d
    return best


def gen_approach(robot, ctx, planner, goal_pos, goal_rotmat, *, tcp, start_qs,
                 chain='main', pre_pos=None, pre_rotmat=None,
                 approach_direction=None, approach_distance=0.05,
                 granularity=0.01, ee_value=None, use_rrt=True,
                 check_descent=True, max_iters=2000, ik_max_solutions=8,
                 ik_accept=None):
    """``start_qs`` -> pre-grasp (probabilistic) -> grasp (cartesian line).

    The pre-grasp is ``goal`` retreated ``approach_distance`` along
    ``-approach_direction`` (default the grasp +z axis) at the grasp orientation,
    unless ``pre_pos`` / ``pre_rotmat`` are given explicitly (e.g. a pre pose
    loaded with the grasp). Stages:

      1. IK the pre-grasp nearest ``start_qs`` that is collision-free (and passes
         ``ik_accept``) -- else None.
      2. ``planner.solve(start_qs, q_pre)`` (RRT) when ``use_rrt``, else a
         collision-gated joint-space line -- else None.
      3. ``omic.linear_to_jpath`` from pre to grasp, seeded for branch
         continuity. ``check_descent`` gates this densified leg with ``ctx``;
         pass False when the grasp INTENTIONALLY contacts the target (the target
         is a collision body, e.g. l1picking) so the contact is not flagged.

    Returns a MotionData (gripper held at ``ee_value`` throughout) ending at the
    grasp config, or None if any stage is infeasible.
    """
    goal_pos = np.asarray(goal_pos, dtype=np.float32)
    goal_rotmat = np.asarray(goal_rotmat, dtype=np.float32)
    if pre_rotmat is None:
        pre_rotmat = goal_rotmat
    if pre_pos is None:
        if approach_direction is None:
            approach_direction = goal_rotmat[:, 2]      # tcp +z = approach axis
        pre_pos = goal_pos - _unit(approach_direction) * float(approach_distance)
    pre_pos = np.asarray(pre_pos, dtype=np.float32)
    pre_rotmat = np.asarray(pre_rotmat, dtype=np.float32)

    q_pre = nearest_valid_ik(robot, ctx, pre_pos, pre_rotmat, chain=chain,
                             tcp=tcp, ref_qs=start_qs,
                             max_solutions=ik_max_solutions, accept=ik_accept)
    if q_pre is None:
        return None

    if use_rrt:
        path = planner.solve(start_qs, q_pre, max_iters=max_iters)
        if not path:
            return None
        travel = MotionData.from_jpath(path, ee_value)
    else:
        seg = omij.linear_path(start_qs, q_pre, ctx=ctx)
        if seg is None:
            return None
        travel = MotionData.from_jpath(seg, ee_value)

    q_seq, _ = omic.linear_to_jpath(
        robot=robot, start_rotmat=pre_rotmat, start_pos=pre_pos,
        goal_rotmat=goal_rotmat, goal_pos=goal_pos, ref_qs=q_pre,
        chain=chain, tcp=tcp, pos_step=granularity,
        ctx=(ctx if check_descent else None))
    if q_seq is None:
        return None
    return travel + MotionData.from_jpath(q_seq, ee_value)


def gen_depart(robot, ctx, planner, start_pos, start_rotmat, *, tcp, start_qs,
               chain='main', depart_direction=None, depart_distance=0.05,
               granularity=0.01, ee_value=None, end_qs=None, use_rrt=False,
               check_retreat=True, max_iters=2000):
    """``start_qs`` (at the grasp) -> retreat (cartesian line) -> optional park.

    A straight cartesian move of ``depart_distance`` along ``depart_direction``
    (default the grasp +z axis -- back out the way the hand came in) at a fixed
    orientation, IK-solved + seeded from ``start_qs``. If ``end_qs`` is given and
    ``use_rrt``, append an RRT from the retreat end to that parking config.
    ``check_retreat`` gates the cartesian leg with ``ctx`` (pass False when still
    in intended contact with the target, mirroring ``gen_approach``).

    Returns a MotionData (gripper held at ``ee_value``) or None if infeasible.
    """
    start_pos = np.asarray(start_pos, dtype=np.float32)
    start_rotmat = np.asarray(start_rotmat, dtype=np.float32)
    if depart_direction is None:
        depart_direction = start_rotmat[:, 2]
    end_pos = start_pos + _unit(depart_direction) * float(depart_distance)

    q_seq, _ = omic.linear_to_jpath(
        robot=robot, start_rotmat=start_rotmat, start_pos=start_pos,
        goal_rotmat=start_rotmat, goal_pos=end_pos, ref_qs=start_qs,
        chain=chain, tcp=tcp, pos_step=granularity,
        ctx=(ctx if check_retreat else None))
    if q_seq is None:
        return None
    md = MotionData.from_jpath(q_seq, ee_value)

    if end_qs is not None and use_rrt:
        path = planner.solve(md.jv_list[-1], np.asarray(end_qs, np.float32),
                             max_iters=max_iters)
        if not path:
            return None
        md = md + MotionData.from_jpath(path, ee_value)
    return md


class ADPlanner:
    """Thin stateful facade over :func:`gen_approach` / :func:`gen_depart`.

    Binds the planning *session* -- ``robot``, the collision ``ctx`` and the
    ``planner`` -- plus default ``chain`` / ``tcp`` once, so call sites pass only
    what actually varies per move (the grasp pose, start config, distances). The
    free functions stay the testable core; this is pure ergonomics over them,
    not a new layer (cf. WRS's ADPlanner, minus the bulk). ``chain`` / ``tcp`` can
    still be overridden per call -- e.g. l1picking's grasp-center tcp changes
    with the jaw width, so it rebuilds the planner per grasp or passes ``tcp=``.
    """

    def __init__(self, robot, ctx, planner, *, chain='main', tcp='flange'):
        self.robot = robot
        self.ctx = ctx
        self.planner = planner
        self.chain = chain
        self.tcp = tcp

    def gen_approach(self, goal_pos, goal_rotmat, *, start_qs,
                     chain=None, tcp=None, **kwargs):
        return gen_approach(
            self.robot, self.ctx, self.planner, goal_pos, goal_rotmat,
            start_qs=start_qs, chain=self.chain if chain is None else chain,
            tcp=self.tcp if tcp is None else tcp, **kwargs)

    def gen_depart(self, start_pos, start_rotmat, *, start_qs,
                   chain=None, tcp=None, **kwargs):
        return gen_depart(
            self.robot, self.ctx, self.planner, start_pos, start_rotmat,
            start_qs=start_qs, chain=self.chain if chain is None else chain,
            tcp=self.tcp if tcp is None else tcp, **kwargs)
