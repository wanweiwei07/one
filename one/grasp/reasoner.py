"""Grasp feasibility reasoning -- the "common grasp" kernel.

Two thin functions, no class, no hidden state:

    find_feasible_gids   -- which grasps are reachable + collision-free at ONE
                            object pose (the per-pose filter).
    reason_common_gids   -- which grasps survive at ALL of several object poses
                            (the intersection -- "shared"/regrasp grasps).

These consolidate the pattern hand-rolled in pick-and-place / shared-grasp demos:
transform each object-local grasp onto the object's world pose, IK the (pre-)grasp
frame, set the gripper to the grasp's opening, and keep the collision-free ones.
Grasps are the (pose, pre_pose, jaw_width, score) tuples from ``antipodal`` /
``load_grasps``; world placement reuses ``serialize.transform_grasps``' convention
(``obj_pose @ local``).

Deliberately scoped to ONE robot against ONE PlanningContext over a set of poses.
Task-specific structure -- dual-arm grasp PAIRS, cable constraints, frozen-other-arm
states (e.g. kurabo) -- stays in the task: it is expressed by how the caller
configures ``ctx`` (and which poses it passes), not baked in here.
"""
import numpy as np


def find_feasible_gids(robot, ctx, grasps, obj_pose, *, tcp, gripper=None,
                       jaw_to_qs=lambda w: (w / 2, w / 2), chain='main',
                       which='pre', max_solutions=1, ik_accept=None):
    """Feasible grasps at a single object pose.

    For each grasp: place its pose (``which='grasp'``) or pre-grasp pose
    (``which='pre'``, the default -- approach reachability) into the world via
    ``obj_pose @ local``, IK it for ``tcp``, set the ``gripper`` to the grasp's
    opening on the collider (so the finger span affects collision), and keep the
    first IK solution that is collision-free under ``ctx`` (and passes the
    optional ``ik_accept(full_qs)`` predicate).

    Parameters
    ----------
    robot, ctx : the arm and its PlanningContext (``ctx.collider`` already holds
        the gripper / obstacles; the caller has set ``ctx`` up for this arm).
    grasps : iterable of (pose, pre_pose, jaw_width, score) -- object-LOCAL.
    obj_pose : (4, 4) object world transform.
    tcp : registered tcp name or TCP object IK solves against.
    gripper : if given, posed to ``jaw_to_qs(jaw_width)`` on ``ctx.collider``
        before each collision check (None skips this -- jaw irrelevant).
    jaw_to_qs : jaw_width -> gripper finger qs (default the symmetric
        parallel-jaw convention, both fingers at width/2).
    which : 'pre' (default) or 'grasp' -- which pose to test for reachability.
    max_solutions : IK solutions to consider per grasp (1 = nearest only).
    ik_accept : optional predicate on the full qs (e.g. a normal-elbow gate).

    Returns
    -------
    dict {gid: qs} -- the feasible grasp index -> its collision-free config.
    """
    obj_pose = np.asarray(obj_pose, dtype=np.float32)
    feasible = {}
    for gid, grasp in enumerate(grasps):
        pose, pre_pose, jaw_width = grasp[0], grasp[1], grasp[2]
        local = pre_pose if which == 'pre' else pose
        world = obj_pose @ np.asarray(local, dtype=np.float32)
        sols = robot.ik(world[:3, 3], world[:3, :3], chain=chain, tcp=tcp,
                        max_solutions=max_solutions)
        if not sols:
            continue
        if gripper is not None:
            ctx.collider.set_mecba_qpos(gripper, jaw_to_qs(jaw_width))
            ctx.clear_cache()
        for s in sols:
            s64 = np.asarray(s, dtype=np.float64)
            if not ctx.is_state_valid(s64):
                continue
            if ik_accept is not None and not ik_accept(s64):
                continue
            feasible[gid] = np.asarray(s, dtype=np.float32)
            break
    return feasible


def reason_common_gids(robot, ctx, grasps, obj_pose_list, **kwargs):
    """Grasps feasible at EVERY object pose in ``obj_pose_list``.

    Runs :func:`find_feasible_gids` per pose and intersects the results -- the
    grasps usable across the whole set (pick AND place, or both bunnies, or a
    regrasp sequence). ``kwargs`` are forwarded to ``find_feasible_gids``.

    Returns
    -------
    dict {gid: [qs_at_pose0, qs_at_pose1, ...]} -- for each common grasp, its
    config at each pose (ordered like ``obj_pose_list``).
    """
    per_pose = [find_feasible_gids(robot, ctx, grasps, p, **kwargs)
                for p in obj_pose_list]
    if not per_pose:
        return {}
    common = set(per_pose[0])
    for d in per_pose[1:]:
        common &= set(d)
    return {gid: [d[gid] for d in per_pose] for gid in sorted(common)}


class GraspReasoner:
    """Thin stateful facade over :func:`find_feasible_gids` /
    :func:`reason_common_gids`.

    Binds the reasoning *session* -- the ``robot``, its collision ``ctx``, the
    reference ``grasps`` and the ``gripper`` / ``tcp`` -- plus the filter
    defaults (``which`` / ``max_solutions`` / ``jaw_to_qs`` / ``ik_accept`` /
    ``chain``) once, so call sites pass only the object pose(s). Mirrors WRS's
    ``GraspReasoner(robot, reference_gc)``; the free functions stay the testable
    core, this is pure ergonomics. Any bound default can be overridden per call
    via keyword (e.g. ``reasoner.find_feasible_gids(pose, which='grasp')``).
    """

    def __init__(self, robot, ctx, grasps, *, tcp, gripper=None, chain='main',
                 jaw_to_qs=lambda w: (w / 2, w / 2), which='pre',
                 max_solutions=1, ik_accept=None):
        self.robot = robot
        self.ctx = ctx
        self.grasps = grasps
        self._defaults = dict(
            tcp=tcp, gripper=gripper, chain=chain, jaw_to_qs=jaw_to_qs,
            which=which, max_solutions=max_solutions, ik_accept=ik_accept)

    def _kw(self, overrides):
        kw = dict(self._defaults)
        kw.update(overrides)
        return kw

    def find_feasible_gids(self, obj_pose, **overrides):
        return find_feasible_gids(self.robot, self.ctx, self.grasps, obj_pose,
                                  **self._kw(overrides))

    def reason_common_gids(self, obj_pose_list, **overrides):
        return reason_common_gids(self.robot, self.ctx, self.grasps,
                                  obj_pose_list, **self._kw(overrides))
