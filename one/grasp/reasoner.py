"""Grasp feasibility reasoning -- the "common grasp" kernel.

Two thin functions, no class, no hidden state:

    find_feasible_gids   -- which grasps are reachable + collision-free at ONE
                            object pose (the per-pose filter).
    reason_common_gids   -- which grasps survive at ALL of several object poses
                            (the intersection -- "shared"/regrasp grasps).

These consolidate the pattern hand-rolled in pick-and-place / shared-grasp demos:
transform each object-local grasp onto the object's world pose, IK the (pre-)grasp
frame, set the gripper to the grasp's frozen qpos, and keep the collision-free
ones. Grasps are :class:`~one.grasp.grasp.Grasp` records from ``antipodal`` /
``load_grasps``; world placement reuses ``serialize.transform_grasps``' convention
(``obj_pose @ local``).

Deliberately scoped to ONE robot against ONE PlanningContext over a set of poses.
Task-specific structure -- dual-arm grasp PAIRS, cable constraints, frozen-other-arm
states (e.g. kurabo) -- stays in the task: it is expressed by how the caller
configures ``ctx`` (and which poses it passes), not baked in here.
"""
import numpy as np


def _feasible_at(robot, ctx, grasp, obj_pose, *, tcp=None, gripper=None,
                 chain='main', which='pre', max_solutions=1, ik_accept=None):
    """One grasp at one object pose -> a collision-free config, or None.

    The single-grasp kernel shared by every reasoning entry point. Places the
    grasp's pre-grasp (``which='pre'``, approach reachability) or grasp
    (``which='grasp'``) pose into the world via ``obj_pose @ local``, IKs it for
    the grasp's frozen tcp, poses the ``gripper`` to the matching frozen qpos on
    the collider (so the finger span affects collision), and returns the first IK
    solution that is collision-free under ``ctx`` and passes the optional
    ``ik_accept(full_qs)`` predicate. Returns None if unreachable / all colliding.
    """
    # the tcp is the grasp's OWN frozen grasp-center frame (no re-derivation from
    # the gripper's current mode / closure) unless one is forced.
    g_tcp = grasp.make_tcp(gripper) if tcp is None else tcp
    local = grasp.pre_pose if which == 'pre' else grasp.pose
    world = obj_pose @ np.asarray(local, dtype=np.float32)
    sols = robot.ik(world[:3, 3], world[:3, :3], chain=chain, tcp=g_tcp,
                    max_solutions=max_solutions)
    if not sols:
        return None
    if gripper is not None:
        qpos = grasp.pre_qpos if which == 'pre' else grasp.qpos
        ctx.collider.set_mecba_qpos(gripper, qpos)
        ctx.clear_cache()
    for s in sols:
        s64 = np.asarray(s, dtype=np.float64)
        if not ctx.is_state_valid(s64):
            continue
        if ik_accept is not None and not ik_accept(s64):
            continue
        return np.asarray(s, dtype=np.float32)
    return None


def iter_common_gids(robot, ctx, grasps, obj_pose_list, *, tcp=None,
                     gripper=None, chain='main', which='pre', max_solutions=1,
                     ik_accept=None):
    """Lazily yield grasps feasible at EVERY pose in ``obj_pose_list``.

    The reasoning CORE: a generator that, per grasp in list order, IKs the first
    pose and -- only if that is feasible -- the next, ..., yielding
    ``(gid, [qs_at_pose0, qs_at_pose1, ...])`` (``gid`` is the index into
    ``grasps``) as soon as the grasp passes ALL poses. Two consequences fall out:

      * a consumer that only needs ONE working grasp (``gen_pick_place``) can
        ``break`` after the first that fully plans -- the remaining grasps are
        never IK'd (the big speedup vs reasoning all of them up front);
      * place-pose IK runs only for grasps already feasible at the pick pose
        (lazy per-pose intersection), no separate intersect step.

    Grasps are visited in LIST order, so to try the likeliest first just pre-sort
    ``grasps`` before calling (``antipodal`` already returns them best-score
    first). Remaining parameters are the per-grasp kernel's (see
    :func:`_feasible_at`).
    """
    if tcp is None and gripper is None:
        raise ValueError("reasoning needs a gripper to host the per-grasp tcp, "
                         "or an explicit tcp")
    poses = [np.asarray(p, dtype=np.float32) for p in obj_pose_list]
    if not poses:
        return
    kw = dict(tcp=tcp, gripper=gripper, chain=chain, which=which,
              max_solutions=max_solutions, ik_accept=ik_accept)
    for gid, grasp in enumerate(grasps):
        qs_list = []
        for pose in poses:
            qs = _feasible_at(robot, ctx, grasp, pose, **kw)
            if qs is None:
                break               # not feasible here -> skip (later poses unIK'd)
            qs_list.append(qs)
        else:                       # every pose passed
            yield gid, qs_list


def find_feasible_gids(robot, ctx, grasps, obj_pose, **kwargs):
    """Feasible grasps at a SINGLE object pose -- the one-pose special case of
    :func:`iter_common_gids`. Returns ``{gid: qs}`` (the collision-free config
    per feasible grasp). ``kwargs`` are the kernel's (``tcp`` / ``gripper`` /
    ``chain`` / ``which`` / ``max_solutions`` / ``ik_accept``)."""
    return {gid: qs_list[0] for gid, qs_list
            in iter_common_gids(robot, ctx, grasps, [obj_pose], **kwargs)}


def reason_common_gids(robot, ctx, grasps, obj_pose_list, **kwargs):
    """All grasps feasible at EVERY pose in ``obj_pose_list`` (the eager form of
    :func:`iter_common_gids`). Returns ``{gid: [qs_at_pose0, ...]}`` -- for each
    common grasp, its config at each pose. Use this when you need the WHOLE
    common set (shared-grasp / regrasp analysis); use ``iter_common_gids`` when
    you can stop at the first that works."""
    return dict(iter_common_gids(robot, ctx, grasps, obj_pose_list, **kwargs))


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

    def __init__(self, robot, ctx, grasps, *, tcp=None, gripper=None,
                 chain='main', which='pre', max_solutions=1, ik_accept=None):
        self.robot = robot
        self.ctx = ctx
        self.grasps = grasps
        self._defaults = dict(
            tcp=tcp, gripper=gripper, chain=chain,
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
