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
``reason_common_gids`` / -- later -- ``regrasp``). It binds NOTHING about the
scene: obstacles are passed per task (``statics=``), so an arm is reusable
across scenes.

Two hosts mix it:
  * a SINGLE-ARM robot mixes it directly -- ``class RS007L(MechBase,
    SingleArmManipulation)`` -- so ``robot.pick_place(...)`` reads naturally and
    no external view wraps the robot. ``body`` is the robot itself.
  * a MULTI-ARM robot OWNS per-arm members built from :class:`Arm`, e.g.
    ``self.left_arm = Arm(self, 'left_arm_waist', self.left_hand)`` -- so
    ``humanoid.left_arm.pick_place(...)``. ``body`` is the parent robot.

The host must provide ``arm_chain`` (the planning chain) and ``end_effector``
(the mounted gripper/hand, assigned after mounting -- swap the EE without
changing the arm). ``body`` is the kinematic robot the arm plans over.
"""
import numpy as np

import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
from one.motion.primitives.approach_depart import gen_approach, gen_depart
from one.grasp.reasoner import reason_common_gids
from one.manipulation.pick_place import gen_pick_place


class SingleArmManipulation:
    """Manipulation verbs for one arm, delegating to the free-function cores.

    Host contract:
      - ``arm_chain``    : str, the planning chain (default ``'main'``).
      - ``end_effector`` : the mounted gripper/hand (set after ``mount``; ``None``
                           until then). Swappable -- same arm, different EE.
      - ``body``         : the kinematic robot to plan over (defaults to ``self``;
                           a multi-arm member overrides it to the parent robot).
    """

    arm_chain = 'main'
    end_effector = None

    @property
    def body(self):
        """The kinematic robot this arm plans over -- ``self`` for a single-arm
        robot; the parent robot for a multi-arm robot's arm member."""
        return self

    # ---- internal: a chain-restricted planning context for this arm + scene ---
    def _planner(self, statics=(), *, margin=0.0, goal_bias=0.3, with_ee=True):
        """Build a (ctx, planner) over ``body`` (+ ``end_effector``) + ``statics``,
        restricted to ``arm_chain`` (other joints frozen). Fresh per call; the
        scene is not bound to the arm."""
        mjc = ocm.MJCollider()
        ents = (self.body, self.end_effector, *statics) if with_ee \
            else (self.body, *statics)
        for e in ents:
            if e is not None:
                mjc.append(e)
        mjc.actors = [self.body]
        mjc.compile(margin=margin, auto_acm=True)
        ctx = omppc.PlanningContext(
            collider=mjc, joint_limits=self.body.chain_joint_limits(self.arm_chain))
        planner = ompr.RRTConnectPlanner(pln_ctx=ctx, goal_bias=goal_bias)
        return ctx, planner

    # ---- grasp reasoning -----------------------------------------------------
    def reason_grasps(self, grasps, obj_pose_list, *, statics=(), margin=0.0,
                      which='pre', **kw):
        """Which object-local ``grasps`` are reachable + collision-free at ALL of
        ``obj_pose_list`` (the shared/regrasp set). Delegates to
        ``reason_common_gids``. Returns ``{gid: [qs_at_pose0, ...]}``."""
        ctx, _ = self._planner(statics, margin=margin)
        return reason_common_gids(self.body, ctx, grasps, obj_pose_list,
                                  gripper=self.end_effector, chain=self.arm_chain,
                                  which=which, **kw)

    # ---- primitive motions ---------------------------------------------------
    def approach(self, goal_pos, goal_rotmat, *, tcp, start_qs=None,
                 statics=(), margin=0.0, **kw):
        """Plan ``start_qs`` -> pre-grasp -> goal (cartesian descent). Delegates
        to ``gen_approach``. ``tcp`` is REQUIRED (the working frame to position --
        there is no universal default tcp across robots); ``str`` is resolved on
        the arm's body."""
        ctx, planner = self._planner(statics, margin=margin)
        if start_qs is None:
            start_qs = np.asarray(self.body.qs, dtype=np.float64).copy()
        if isinstance(tcp, str):
            tcp = self.body.tcp(tcp)
        return gen_approach(self.body, ctx, planner, goal_pos, goal_rotmat,
                            tcp=tcp, start_qs=start_qs, chain=self.arm_chain, **kw)

    def depart(self, start_pos, start_rotmat, *, tcp, start_qs,
               statics=(), margin=0.0, **kw):
        """Retreat from a grasp along the approach axis (cartesian). Delegates to
        ``gen_depart``. ``tcp`` is REQUIRED (``str`` resolved on the body)."""
        ctx, planner = self._planner(statics, margin=margin)
        if isinstance(tcp, str):
            tcp = self.body.tcp(tcp)
        return gen_depart(self.body, ctx, planner, start_pos, start_rotmat,
                          tcp=tcp, start_qs=start_qs, chain=self.arm_chain, **kw)

    # ---- pick & place --------------------------------------------------------
    def pick_place(self, obj, grasps, pick_pose, place_pose, *, statics=(), **kw):
        """Full home -> pick -> lift -> transfer -> place -> retreat for ``obj``
        from ``pick_pose`` to ``place_pose``. Delegates to ``gen_pick_place``
        (which reasons a grasp feasible at both poses and builds its own two-phase
        colliders). ``grasps`` are object-LOCAL and must be planned for THIS
        ``end_effector`` (gen_pick_place guards the binding)."""
        return gen_pick_place(self.body, self.end_effector, obj, grasps,
                              pick_pose, place_pose, statics=statics,
                              chain=self.arm_chain, **kw)

    # ---- regrasp (FUTURE) ----------------------------------------------------
    def regrasp(self, obj, grasps, placements, start_pose, goal_pose, *,
                statics=(), **kw):
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
    ``body``, one ``arm_chain``, and one ``end_effector``. Created by the robot
    in its ``__init__`` (e.g. ``self.left_arm = Arm(self, 'left_arm_waist',
    self.left_hand)``) -- it is the robot's own arm, not an external wrapper.

    (A single-arm robot does NOT use this: it mixes ``SingleArmManipulation``
    directly and IS its own arm.)"""

    def __init__(self, body, arm_chain, end_effector=None):
        self._body = body
        self.arm_chain = arm_chain
        self.end_effector = end_effector

    @property
    def body(self):
        return self._body
