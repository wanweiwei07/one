"""Sequential handover: Left UR3+KRBLeft places con_ma at a fixed mid-
air handover pose, retreats; Right UR3+KRBRight then grasps it and
places it at the destination.

Phase order:
    A   left home -> pick (right at home)
    A'  left close + attach
    B   left -> handover (carrying)
    B'  left release (con freezes at handover pose)
    C   left -> home (right still at home)
    D   right home -> grasp at handover
    D'  right close + attach
    E   right -> place
    E'  right release; con snaps to PLACE_POS
    F   right -> home
"""
import os
import sys
import builtins
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.collider.mj_collider as ocm
import one.utils.constant as ouc
import one.utils.math as oum
import one.viewer.world as ovw
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.motion.probabilistic.planning_context as omppc
import one.motion.probabilistic.prm as ompp
from one.grasp.antipodal import antipodal

from kurabo.robot import LeftRobot, RightRobot


LEFT_BASE = np.array([0.0, -0.3, 0.0], dtype=np.float32)
RIGHT_BASE = np.array([0.0, 0.3, 0.0], dtype=np.float32)
PICK_POS = np.array([0.2, -0.25, 0.05], dtype=np.float32)
PLACE_POS = np.array([0.2, 0.25, 0.05], dtype=np.float32)
HANDOVER_POS = np.array([0.2, 0.0, 0.3], dtype=np.float32)
PARK_POS = np.array([5.0, 5.0, 5.0], dtype=np.float32)  # off-scene parking
IDENTITY = np.eye(3, dtype=np.float32)


def main():
    base = ovw.World(cam_pos=(1.6, 1.2, 1.0),
                     cam_lookat_pos=(0.1, 0.0, 0.25))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    lft = LeftRobot(pos=LEFT_BASE)
    rgt = RightRobot(pos=RIGHT_BASE)
    for r in (lft, rgt):
        r.attach_to(base.scene)
        r.gripper.attach_to(base.scene)
    # Show collision geometry during animation.
    lft.toggle_render_collision = True
    rgt.toggle_render_collision = True
    lft.gripper.toggle_render_collision = True
    rgt.gripper.toggle_render_collision = True
    # Ground plane (slightly below z=0 to avoid base-vs-ground self-hit)
    ground = ossop.plane(pos=(0, 0, -0.001))
    ground.attach_to(base.scene)

    obj_path = os.path.join(_PROJECT_ROOT, "kurabo", "objs", "con_ma.stl")
    con = osso.SceneObject.from_file(
        obj_path, collision_type=ouc.CollisionType.MESH, is_free=True)
    con.rgb = (0.85, 0.6, 0.3)
    # Run antipodal at world origin so returned poses are in object-local
    # frame (later we post-multiply by tf_obj to get world poses).
    con.pos = np.zeros(3, dtype=np.float32)
    con.rotmat = IDENTITY
    con.attach_to(base.scene)

    ossop.frame(pos=HANDOVER_POS).attach_to(base.scene)
    ossop.frame(pos=PLACE_POS).attach_to(base.scene)

    print("Computing antipodal grasps on con_ma...")
    grasps = antipodal(
        gripper=lft.gripper, target_sobj=con,
        density=0.003, normal_tol_deg=25,
        roll_step_deg=20, clearance=0.0001,
        max_grasps=300)
    print(f"  found {len(grasps)} grasps")
    if not grasps:
        print("No grasps. Aborting.")
        base.run()
        return

    # now move con to the actual pick pose
    con.pos = PICK_POS
    con.rotmat = IDENTITY

    # con is intentionally not appended to mjc: mujoco approximates meshes
    # with their convex hull and would falsely report finger-vs-con
    # penetrations even at antipodal-verified grasp poses. Antipodal
    # already validates collisions for the actual grasp.
    mjc = ocm.MJCollider()
    for body in (lft, lft.gripper, rgt, rgt.gripper, ground):
        mjc.append(body)
    mjc.actors = [lft, rgt]  # set both as actors for IK validation
    mjc.compile(margin=0.0)

    sl_lft = mjc.get_slice(lft)
    sl_rgt = mjc.get_slice(rgt)

    # Single combined context (both arms as actors) used during IK +
    # grasp-feasibility checks where we need to evaluate full 12-DOF
    # states.
    pln_ctx = omppc.PlanningContext(collider=mjc, cd_step_size=np.pi / 90)

    # Per-arm contexts for motion planning: only the active arm is an
    # actor (6-DOF state space); the other arm + both grippers are
    # frozen via set_aux_mecbas. This keeps the planner search space
    # tight (6 DOF instead of 12).
    def _build_arm_ctx(active_arm):
        c = ocm.MJCollider()
        for body in (lft, lft.gripper, rgt, rgt.gripper, ground):
            c.append(body)
        c.actors = [active_arm]
        c.compile(margin=0.0)
        ctx = omppc.PlanningContext(collider=c, cd_step_size=np.pi / 90)
        return ctx

    pln_ctx_lft = _build_arm_ctx(lft)
    pln_ctx_rgt = _build_arm_ctx(rgt)
    planner_lft = ompp.LazyPRMPlanner(
        pln_ctx=pln_ctx_lft, k=15, n_samples=500)
    planner_rgt = ompp.LazyPRMPlanner(
        pln_ctx=pln_ctx_rgt, k=15, n_samples=500)
    PRM_SEED = 42

    home_lft = lft.qs.copy()
    home_rgt = rgt.qs.copy()
    open_jaw = float(lft.gripper.jaw_range[1])

    def state(lq, rq):
        out = np.zeros(lft.ndof + rgt.ndof, dtype=np.float32)
        out[sl_lft] = lq
        out[sl_rgt] = rq
        return out

    def set_jaws(jaw_l, jaw_r):
        pln_ctx.set_aux_mecbas(
            lft.gripper, qs=(jaw_l / 2, jaw_l / 2))
        pln_ctx.set_aux_mecbas(
            rgt.gripper, qs=(jaw_r / 2, jaw_r / 2))

    def set_obj(pos, rot=IDENTITY):
        con.pos = np.asarray(pos, dtype=np.float32)
        con.rotmat = np.asarray(rot, dtype=np.float32)

    def find_arm_grasp(arm, target_pos, target_rot,
                       other_arm_qs, other_jaw,
                       this_jaw=None, approach_dir=None,
                       approach_dot_min=0.5):
        """Find one grasp whose actual grasp-pose IK is valid for `arm`
        (TCP at the contact midpoint, not at pre-grasp), without
        colliding with `other_arm` frozen at `other_arm_qs`. Tries every
        IK branch per grasp candidate. `approach_dir` (3-vec in world)
        filters grasps whose gripper-Z (approach axis) aligns with it.

        During collision validation jaws are held fully open so that
        mujoco's convex-hull approximation of the object doesn't produce
        spurious finger penetrations (antipodal already verified that
        the fingers fit at the actual `jaw_w`)."""
        tf_obj = oum.tf_from_rotmat_pos(target_rot, target_pos)
        approach_dir = (None if approach_dir is None
                        else np.asarray(approach_dir, dtype=np.float32))
        for gi, (pose, _pre, jaw_w, _) in enumerate(grasps):
            if this_jaw is not None and abs(jaw_w - this_jaw) > 1e-6:
                continue
            gl_pose = tf_obj @ pose
            if approach_dir is not None:
                if float(gl_pose[:3, 2] @ approach_dir) < approach_dot_min:
                    continue
            qs_list = arm.ik_tcp(
                tgt_rotmat=gl_pose[:3, :3], tgt_pos=gl_pose[:3, 3])
            if not qs_list:
                continue
            set_jaws(open_jaw, open_jaw)
            for qs in qs_list:
                if arm is lft:
                    full = state(qs, other_arm_qs)
                else:
                    full = state(other_arm_qs, qs)
                if pln_ctx.is_state_valid(full):
                    return gi, qs, jaw_w
        return None, None, None

    # During grasp-pose IK validation, park con far away so mujoco's
    # convex-hull approximation of con doesn't trigger spurious finger
    # penetrations (antipodal already verified the grasp itself).
    # ---- IK for all key configurations ----
    # Pick + handover for the LEFT arm: search jointly so the same grasp
    # index is IK-feasible at PICK_POS AND HANDOVER_POS (the gripper
    # holds the object the same way throughout).
    print("Solving left @ pick + handover (joint search)...")
    qs_lft_pick = qs_lft_ho = jaw_pick = None
    tf_pick = oum.tf_from_rotmat_pos(IDENTITY, PICK_POS)
    tf_ho = oum.tf_from_rotmat_pos(IDENTITY, HANDOVER_POS)
    set_obj(PARK_POS)
    set_jaws(open_jaw, open_jaw)
    for pose_g, _pre, jw, _ in grasps:
        gl_pick = tf_pick @ pose_g
        gl_ho = tf_ho @ pose_g
        ql_pick = lft.ik_tcp(
            tgt_rotmat=gl_pick[:3, :3], tgt_pos=gl_pick[:3, 3])
        if not ql_pick:
            continue
        ql_ho = lft.ik_tcp(
            tgt_rotmat=gl_ho[:3, :3], tgt_pos=gl_ho[:3, 3])
        if not ql_ho:
            continue
        q_pick = next(
            (q for q in ql_pick
             if pln_ctx.is_state_valid(state(q, home_rgt))), None)
        if q_pick is None:
            continue
        q_ho = next(
            (q for q in ql_ho
             if pln_ctx.is_state_valid(state(q, home_rgt))), None)
        if q_ho is None:
            continue
        qs_lft_pick = q_pick
        qs_lft_ho = q_ho
        jaw_pick = jw
        break
    assert qs_lft_pick is not None, "no left grasp valid at both pick & handover"

    # Right @ handover: con is alone (left has retreated to home).
    # Park con during validation so mujoco's hull doesn't false-collide.
    print("Solving right @ handover...")
    set_obj(PARK_POS)
    _, qs_rgt_ho, jaw_ho_rgt = find_arm_grasp(
        rgt, HANDOVER_POS, IDENTITY,
        other_arm_qs=home_lft, other_jaw=open_jaw)
    assert qs_rgt_ho is not None, "right IK at handover failed"

    # Right @ place: con parked during validation (we'll set its visual
    # pose from the animation callback when right releases).
    set_obj(PARK_POS)
    print("Solving right @ place...")
    _, qs_rgt_place, jaw_place = find_arm_grasp(
        rgt, PLACE_POS, IDENTITY,
        other_arm_qs=home_lft, other_jaw=open_jaw)
    assert qs_rgt_place is not None, "right IK at place failed"

    # ---- Plan segments (per-arm, 6-DOF) ----
    # Each segment plans only the active arm. The inactive arm is
    # frozen via set_aux_mecbas; both grippers are also aux.
    def plan_arm(active, start_q, goal_q, jaw_l, jaw_r,
                 frozen_other_qs):
        if active is lft:
            ctx, plnr, other = pln_ctx_lft, planner_lft, rgt
        else:
            ctx, plnr, other = pln_ctx_rgt, planner_rgt, lft
        ctx.set_aux_mecbas(other, qs=frozen_other_qs)
        ctx.set_aux_mecbas(lft.gripper, qs=(jaw_l / 2, jaw_l / 2))
        ctx.set_aux_mecbas(rgt.gripper, qs=(jaw_r / 2, jaw_r / 2))
        # Seed before each PRM build so sampling is reproducible.
        np.random.seed(PRM_SEED)
        path = plnr.solve(start=start_q, goal=goal_q)
        if not path:
            raise RuntimeError("planning failed")
        return path

    print("Planning A (left home -> pick)...")
    pathA_q = plan_arm(lft, home_lft, qs_lft_pick,
                       open_jaw, open_jaw, frozen_other_qs=home_rgt)

    print("Planning B (left pick -> handover, carrying)...")
    pathB_q = plan_arm(lft, qs_lft_pick, qs_lft_ho,
                       jaw_pick, open_jaw, frozen_other_qs=home_rgt)

    print("Planning C (left -> home)...")
    pathC_q = plan_arm(lft, qs_lft_ho, home_lft,
                       open_jaw, open_jaw, frozen_other_qs=home_rgt)

    print("Planning D (right -> handover)...")
    pathD_q = plan_arm(rgt, home_rgt, qs_rgt_ho,
                       open_jaw, open_jaw, frozen_other_qs=home_lft)

    print("Planning E (right -> place, carrying)...")
    pathE_q = plan_arm(rgt, qs_rgt_ho, qs_rgt_place,
                       open_jaw, jaw_ho_rgt, frozen_other_qs=home_lft)

    print("Planning F (right -> home)...")
    pathF_q = plan_arm(rgt, qs_rgt_place, home_rgt,
                       open_jaw, open_jaw, frozen_other_qs=home_lft)

    # Convert each per-arm path to a list of 12-DOF full states for the
    # animation tick (uses sl_lft / sl_rgt to read from full state).
    def expand(active, qs_path, frozen_other_qs):
        out = []
        for q in qs_path:
            if active is lft:
                full = state(q, frozen_other_qs)
            else:
                full = state(frozen_other_qs, q)
            out.append(full)
        return out

    pathA = expand(lft, pathA_q, home_rgt)
    pathB = expand(lft, pathB_q, home_rgt)
    pathC = expand(lft, pathC_q, home_rgt)
    pathD = expand(rgt, pathD_q, home_lft)
    pathE = expand(rgt, pathE_q, home_lft)
    pathF = expand(rgt, pathF_q, home_lft)

    print("All plans OK. Animating.")

    # Reset to start for animation
    lft.fk(qs=home_lft)
    rgt.fk(qs=home_rgt)
    lft.gripper.set_jaw_width(open_jaw)
    rgt.gripper.set_jaw_width(open_jaw)
    set_obj(PICK_POS)

    # Phases for animation: each is (path, on_enter)
    def attach_left():
        lft.gripper.set_jaw_width(jaw_pick)
        lft.gripper.grasp(con, jaw_width=jaw_pick)
        print("[anim] left attached con")

    def release_left_at_handover():
        lft.gripper.release(con, jaw_width=open_jaw)
        # freeze con at the handover pose
        con.pos = HANDOVER_POS
        con.rotmat = IDENTITY
        print("[anim] left released con at handover")

    def attach_right():
        rgt.gripper.set_jaw_width(jaw_ho_rgt)
        rgt.gripper.grasp(con, jaw_width=jaw_ho_rgt)
        print("[anim] right attached con")

    def release_right_at_place():
        rgt.gripper.release(con, jaw_width=open_jaw)
        con.pos = PLACE_POS
        con.rotmat = IDENTITY
        print("[anim] right released con at place")

    phase_paths = [
        (pathA, None),                           # left -> pick
        (pathB, attach_left),                    # close, carry to handover
        (pathC, release_left_at_handover),       # release, retreat to home
        (pathD, None),                           # right -> handover
        (pathE, attach_right),                   # close, carry to place
        (pathF, release_right_at_place),         # release, retreat to home
    ]

    phase_idx = [0]
    cursor = [0]
    entered = [False]

    def tick(dt):
        if phase_idx[0] >= len(phase_paths):
            return
        path, on_enter = phase_paths[phase_idx[0]]
        if not entered[0]:
            if on_enter is not None:
                on_enter()
            entered[0] = True
        if cursor[0] < len(path):
            s = path[cursor[0]]
            lft.fk(qs=s[sl_lft])
            rgt.fk(qs=s[sl_rgt])
            cursor[0] += 1
        else:
            phase_idx[0] += 1
            cursor[0] = 0
            entered[0] = False

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == "__main__":
    main()
