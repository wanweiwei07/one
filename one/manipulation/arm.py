"""The manipulation verbs of a single arm -- as a mixin, not a wrapper.

Ontology (see also the discussion that produced this):
  * **arm == manipulator** -- the SAME object, two aspects: "arm" = reach
    (a kinematic chain + ik), "manipulator" = capability (approach / pick_place /
    regrasp). A single-arm robot IS its own arm; a multi-arm robot's
    ``left_arm`` / ``right_arm`` each IS an arm.
  * **end_effector** -- a SEPARATE, swappable device the arm carries (mounted at
    its tip): a gripper / hand / tool. The arm reaches; the end_effector grasps.

So manipulation = arm (reach) + end_effector (grasp) + behavior. This module
provides the behavior as ``SingleArmManipulation`` (a mixin) that delegates to
the free-function cores (``gen_approach`` / ``gen_depart`` / ``gen_pick_place`` /
``reason_common_gids`` / -- later -- ``regrasp``). The scene is a collision world
built once via ``collider = arm.build_collider(fixtures=..., objects=...)`` and
PASSED explicitly to each planning call (``collider=``) -- it is a value, not
hidden state, so a call site shows which world it plans in and you cannot forget
it. Each method reuses the passed collider (context cached by its identity), so
there is no per-call rebuild; only a grasp / release ``refresh``es it.

Two hosts mix it:
  * a SINGLE-ARM robot mixes it directly -- ``class RS007L(MechBase,
    SingleArmManipulation)`` -- so ``robot.pick_place(...)`` reads naturally and
    no external view wraps the robot. ``body`` is the robot itself.
  * a MULTI-ARM robot OWNS per-arm members built from :class:`Arm`, e.g.
    ``self.left_arm = Arm(self, 'left_arm_waist')`` -- so
    ``humanoid.left_arm.pick_place(...)``. ``body`` is the parent robot.

The host provides ``arm_chain`` (the planning chain); ``end_effector`` is DERIVED
from the mech mounted at the chain tip (named once -- when mounted -- and swap by
re-mounting), set explicitly only to disambiguate several EEs at the flange.
``body`` is the kinematic robot the arm plans over.
"""
import numpy as np

import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.motion.primitives.approach_depart import (
    gen_approach, gen_depart, gen_moveto)
from one.grasp.reasoner import reason_common_gids
from one.manipulation.pick_place import gen_pick_place, iter_pick_place


class SingleArmManipulation:
    """Manipulation verbs for one arm, delegating to the free-function cores.

    Host contract:
      - ``arm_chain``    : str, the planning chain (default ``'main'``).
      - ``end_effector`` : the gripper/hand at the arm's flange. DERIVED from the
                           mount (the mech mounted at the chain tip), so it is
                           named once -- when mounted -- and swapping it (re-mount)
                           just works. Set it explicitly (``arm.end_effector = ee``
                           or ``Arm(..., ee)``) only to disambiguate when several
                           EEs sit at the flange.
      - ``body``         : the kinematic robot to plan over (defaults to ``self``;
                           a multi-arm member overrides it to the parent robot).
    """

    arm_chain = 'main'
    _ee_override = None

    @property
    def body(self):
        """The kinematic robot this arm plans over -- ``self`` for a single-arm
        robot; the parent robot for a multi-arm robot's arm member."""
        return self

    @property
    def end_effector(self):
        """The mech mounted at this arm's flange (the chain tip). An explicit
        override wins; otherwise it is DERIVED from the mount -- ``None`` if
        nothing is mounted there, and a loud error if SEVERAL are (then set it
        explicitly). Read a handful of times per planning op, not in a hot loop."""
        if self._ee_override is not None:
            return self._ee_override
        tip_lidx = self.body.chain(self.arm_chain).tip_lidx
        found = [c for c, m in self.body._mountings.items()
                 if hasattr(c, 'runtime_lnks')
                 and self.body.runtime_lidx_map.get(m.plnk) == tip_lidx]
        if len(found) > 1:
            raise RuntimeError(
                f"{len(found)} end effectors at the '{self.arm_chain}' flange; "
                f"set arm.end_effector explicitly to choose one")
        return found[0] if found else None

    @end_effector.setter
    def end_effector(self, ee):
        self._ee_override = ee

    # ---- the collision world -------------------------------------------------
    def build_collider(self, fixtures=(), objects=(), *, margin=0.0):
        """Build + compile a collision world for this arm and RETURN it -- the
        caller holds it and passes it (as ``collider=``) to each planning method.
        It is an explicit VALUE, never hidden state on the robot, so a call site
        always shows which world it plans in and you cannot forget to set it up.
        Contents: ``robot`` (+ its mounted ``end_effector``) + static ``fixtures``
        + obstacle ``objects``. Do NOT include the object you are about to pick --
        it is grasped (mounted on the gripper), not an obstacle. Reused across
        every call it is passed to (only a grasp / release ``refresh``es it); for
        a multi-arm robot one collider serves both arms."""
        c = ocm.MJCollider(self.body, fixtures=fixtures, objects=objects)
        c.compile(margin=margin, auto_acm=True)
        return c

    # ---- recipe composition --------------------------------------------------
    def plan(self, collider, *, constraints=(), tcp=None, start_qs=None):
        """Open a manipulation :class:`~one.manipulation.session.Session` bound to
        this arm + ``collider`` + ``constraints``. List steps (``moveto`` /
        ``linear`` / ``grasp`` / ``release``) then read ``.result`` (a MotionData
        or None). The thin, recommended way to write a multi-step recipe; for a
        multi-arm cell use ``Workcell.plan`` (it activates the arm first)."""
        from one.manipulation.session import Session
        return Session(self, collider, constraints=constraints, tcp=tcp,
                       start_qs=start_qs)

    def _context(self, collider, constraints=()):
        """(ctx, planner) over ``collider``, restricted to this arm's chain, with
        optional extra ``constraints`` (a tuple of
        :class:`~one.motion.core.constraint.Constraint` -- e.g. a cable/tether
        limit). Cached keyed by the collider IDENTITY and the constraint set, so
        repeated calls with the same world + constraints reuse one context."""
        constraints = tuple(constraints)
        cache = getattr(self, '_ctx_cache', None)
        if (cache is None or cache[0] is not collider
                or cache[3] != constraints):
            ctx = omppc.PlanningContext(
                collider=collider,
                joint_limits=self.body.chain_joint_limits(self.arm_chain),
                constraints=constraints)
            planner = ompr.RRTConnectPlanner(pln_ctx=ctx, goal_bias=0.3)
            self._ctx_cache = (collider, ctx, planner, constraints)
        return self._ctx_cache[1], self._ctx_cache[2]

    # ---- grasp reasoning -----------------------------------------------------
    def reason_grasps(self, grasps, obj_pose_list, *, collider, constraints=(),
                      which='pre', **kw):
        """Which object-local ``grasps`` are reachable + collision-free (and
        ``constraints``-satisfying) at ALL of ``obj_pose_list`` (the shared/regrasp
        set) in ``collider``. Delegates to ``reason_common_gids``. Returns
        ``{gid: [qs_at_pose0, ...]}``."""
        ctx, _ = self._context(collider, constraints)
        return reason_common_gids(self.body, ctx, grasps, obj_pose_list,
                                  gripper=self.end_effector, chain=self.arm_chain,
                                  which=which, **kw)

    # ---- primitive motions ---------------------------------------------------
    def moveto(self, goal, *, collider, tcp=None, constraints=(), start_qs=None,
               ee_qpos=None, max_iters=4000):
        """Free RRT move to ``goal`` -- a full joint config, or a tcp pose
        (``(pos, rotmat)`` / 4x4) IK'd via ``tcp`` -- in ``collider``, gated by
        ``constraints``. From ``start_qs`` (default: current). Returns a
        MotionData (held at ``ee_qpos``) or None. The free-space primitive that
        ``moveto``-style legs and the :class:`Session` build on; a ``str`` ``tcp``
        is resolved on the body."""
        ctx, planner = self._context(collider, constraints)
        ctx.clear_cache()   # collider state (frozen aux arms, jaws) may have changed
        if start_qs is None:
            start_qs = np.asarray(self.body.qs, dtype=np.float64).copy()
        if isinstance(tcp, str):
            tcp = self.body.tcp(tcp)
        return gen_moveto(self.body, ctx, planner, goal, tcp=tcp,
                          start_qs=start_qs, chain=self.arm_chain,
                          ee_qpos=ee_qpos, max_iters=max_iters)

    def approach(self, goal_pos, goal_rotmat, *, collider, tcp, constraints=(),
                 start_qs=None, **kw):
        """Plan ``start_qs`` -> pre-grasp -> goal (cartesian descent) in
        ``collider``, gated by optional extra ``constraints`` (cable/tether/keep-
        out predicates). Delegates to ``gen_approach``. ``collider`` and ``tcp``
        are REQUIRED (the world to plan in, and the working frame to position --
        no universal default tcp across robots); a ``str`` ``tcp`` is resolved on
        the arm's body."""
        ctx, planner = self._context(collider, constraints)
        ctx.clear_cache()   # collider state (frozen aux arms, jaws) may have changed
        if start_qs is None:
            start_qs = np.asarray(self.body.qs, dtype=np.float64).copy()
        if isinstance(tcp, str):
            tcp = self.body.tcp(tcp)
        return gen_approach(self.body, ctx, planner, goal_pos, goal_rotmat,
                            tcp=tcp, start_qs=start_qs, chain=self.arm_chain, **kw)

    def depart(self, start_pos, start_rotmat, *, collider, tcp, start_qs,
               constraints=(), **kw):
        """Retreat from a grasp along the approach axis (cartesian) in
        ``collider``, gated by optional extra ``constraints``. Delegates to
        ``gen_depart``. ``collider`` and ``tcp`` are REQUIRED (``str`` ``tcp``
        resolved on the body)."""
        ctx, planner = self._context(collider, constraints)
        ctx.clear_cache()   # collider state (frozen aux arms, jaws) may have changed
        if isinstance(tcp, str):
            tcp = self.body.tcp(tcp)
        return gen_depart(self.body, ctx, planner, start_pos, start_rotmat,
                          tcp=tcp, start_qs=start_qs, chain=self.arm_chain, **kw)

    def insert(self, goal_pos, goal_rotmat, *, collider, tcp, start_qs,
               constraints=(), start_pos=None, start_rotmat=None,
               granularity=0.005, **kw):
        """Straight CARTESIAN move (no RRT) of ``tcp`` to ``(goal_pos,
        goal_rotmat)`` in ``collider``, gated by ``constraints`` -- the mating /
        insertion leg (pure translation along the working axis at a fixed pose).
        Starts at ``(start_pos, start_rotmat)`` if given, else at the current
        ``tcp`` of ``start_qs``; per-step IK is seeded from ``start_qs`` and each
        densified edge is gated. Returns a MotionData (or None). Delegates to
        ``gen_approach`` with ``use_rrt=False`` (a pure cartesian line)."""
        ctx, planner = self._context(collider, constraints)
        ctx.clear_cache()   # collider state (frozen aux arms, jaws) may have changed
        if isinstance(tcp, str):
            tcp = self.body.tcp(tcp)
        if start_pos is None or start_rotmat is None:
            self.body.fk(qs=start_qs)
            cur = np.asarray(tcp.tf, dtype=np.float32)
            start_pos = cur[:3, 3] if start_pos is None else start_pos
            start_rotmat = cur[:3, :3] if start_rotmat is None else start_rotmat
        return gen_approach(self.body, ctx, planner, goal_pos, goal_rotmat,
                            tcp=tcp, start_qs=start_qs, chain=self.arm_chain,
                            pre_pos=start_pos, pre_rotmat=start_rotmat,
                            granularity=granularity, use_rrt=False,
                            check_descent=True, **kw)

    # ---- holding -------------------------------------------------------------
    def grasp(self, obj, qpos, *, collider, exclude=False):
        """Take ``obj`` into this arm's end effector: mount it at the EE's grasp,
        pose the EE to ``qpos`` (the closed config), and ``refresh`` ``collider``
        so later carry legs account for it. The held object rides the gripper and
        is collision-checked by default; pass ``exclude=True`` to keep it OUT of
        collision (e.g. a connector whose clearance is enforced by a cable
        constraint instead). ``release`` undoes it."""
        ee = self.end_effector
        ee.attach(obj, qpos)
        if exclude:
            collider.exclude(obj)
        collider.refresh()
        collider.set_mecba_qpos(ee, np.asarray(qpos, dtype=np.float32))

    def release(self, obj, *, collider):
        """Drop ``obj`` from the end effector and ``refresh`` ``collider`` back to
        the free state (the inverse of :meth:`grasp`)."""
        ee = self.end_effector
        ee.detach(obj)
        collider.include(obj)
        collider.refresh()

    # ---- pick & place --------------------------------------------------------
    def pick_place(self, obj, grasps, pick_pose, place_pose, *, collider, **kw):
        """Full home -> pick -> lift -> transfer -> place -> retreat for ``obj``
        from ``pick_pose`` to ``place_pose`` in ``collider`` (REQUIRED; it must
        NOT contain ``obj`` -- the object is grasped, not an obstacle). Delegates
        to ``gen_pick_place``. ``grasps`` are object-LOCAL and must be planned for
        THIS ``end_effector`` (gen_pick_place guards the binding)."""
        return gen_pick_place(self.body, self.end_effector, obj, grasps,
                              pick_pose, place_pose, collider=collider,
                              chain=self.arm_chain, **kw)

    def iter_pick_place(self, obj, grasps, pick_pose, place_pose, *, collider,
                        **kw):
        """Lazy form of :meth:`pick_place`: a generator yielding ``(gid,
        MotionData)`` for EACH common grasp that fully plans, planned on demand.
        ``pick_place`` is just its first item; a viewer can pull the next grasp's
        plan when the user asks. Same contract as ``pick_place`` (``collider``
        REQUIRED, must NOT contain ``obj``; ``grasps`` planned for THIS
        ``end_effector``)."""
        return iter_pick_place(self.body, self.end_effector, obj, grasps,
                               pick_pose, place_pose, collider=collider,
                               chain=self.arm_chain, **kw)

    # ---- regrasp (FUTURE) ----------------------------------------------------
    def regrasp(self, obj, grasps, placements, start_pose, goal_pose, **kw):
        """Multi-step regrasp from ``start_pose`` to ``goal_pose`` via
        intermediate ``placements`` when no single grasp is feasible at both.

        Planned shape: a combinatorial regrasp GRAPH (nodes = (placement, grasp,
        this arm); edges = transfer [carry, same grasp, two placements -> one
        ``pick_place`` leg] / transit [regrasp in place, two grasps -> ``depart``
        + ``approach``]); search -> realize the path into a ``MotionData`` (the
        ``+``-composition of edge motions); prune+re-search on motion failure.
        The 0-regrasp path is exactly ``pick_place``. See one/manipulation/
        regrasp/ (to be added)."""
        raise NotImplementedError("regrasp graph -- see module docstring roadmap")


class Arm(SingleArmManipulation):
    """A per-arm manipulator OWNED by a multi-arm robot: binds the parent
    ``body`` and one ``arm_chain``. Created by the robot in its ``__init__``
    (e.g. ``self.left_arm = Arm(self, 'left_arm_waist')``) -- it is the robot's
    own arm, not an external wrapper. The ``end_effector`` is DERIVED from what is
    mounted at the chain tip; pass it only to disambiguate several EEs there.

    (A single-arm robot does NOT use this: it mixes ``SingleArmManipulation``
    directly and IS its own arm.)"""

    def __init__(self, body, arm_chain, end_effector=None):
        self._body = body
        self.arm_chain = arm_chain
        self.end_effector = end_effector       # None -> derive from the mount

    @property
    def body(self):
        return self._body
