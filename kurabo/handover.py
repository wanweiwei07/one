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
from one.grasp.serialize import load_grasps

from kurabo.robot import LeftRobot, RightRobot
from kurabo import grasp as kgrasp


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

    obj_path = os.path.join(
        _PROJECT_ROOT, "kurabo", "objects", "con_ma.stl")
    con = osso.SceneObject.from_file(
        obj_path, collision_type=ouc.CollisionType.MESH, is_free=True)
    con.rgb = (0.85, 0.6, 0.3)
    con.pos = PICK_POS
    con.rotmat = IDENTITY
    con.attach_to(base.scene)

    ossop.frame(pos=HANDOVER_POS).attach_to(base.scene)
    ossop.frame(pos=PLACE_POS).attach_to(base.scene)

    # Grasps are pre-computed in object-local frame by kurabo/grasp.py.
    if not os.path.isfile(kgrasp.OUT_PATH):
        print(f"Grasp dump {kgrasp.OUT_PATH} missing.")
        print("Run `python kurabo/grasp.py` first.")
        return
    print(f"Loading grasps from {kgrasp.OUT_PATH}...")
    grasps = load_grasps(kgrasp.OUT_PATH)
    print(f"  loaded {len(grasps)} grasps")
    if not grasps:
        print("No grasps. Aborting.")
        base.run()
        return

    # con is intentionally not appended to mjc: mujoco approximates meshes
    # with their convex hull and would falsely report finger-vs-con
    # penetrations even at antipodal-verified grasp poses. Antipodal
    # already validates collisions for the actual grasp.
    mjc = ocm.MJCollider()
    for body in (lft, lft.gripper, rgt, rgt.gripper, ground):
        mjc.append(body)
    mjc.actors = [lft, rgt]  # set both as actors for IK validation
    mjc.compile(margin=0.0)

    # Per-arm planning contexts share the same MJCollider (compile/MJCF
    # parsing is the expensive part; PlanningContext init is cheap).
    # We flip `mjc.actors` to the active arm right before each plan call
    # so the corresponding context's slice mapping is correct.
    pln_ctx_lft = omppc.PlanningContext(
        collider=mjc, planning_mecbas=[lft], cd_step_size=np.pi / 90)
    pln_ctx_rgt = omppc.PlanningContext(
        collider=mjc, planning_mecbas=[rgt], cd_step_size=np.pi / 90)
    planner_lft = ompp.LazyPRMPlanner(
        pln_ctx=pln_ctx_lft, k=15, n_samples=500)
    planner_rgt = ompp.LazyPRMPlanner(
        pln_ctx=pln_ctx_rgt, k=15, n_samples=500)
    PRM_SEED = 42

    home_lft = lft.qs.copy()
    home_rgt = rgt.qs.copy()
    open_jaw = float(lft.gripper.jaw_range[1])

    def set_obj(pos, rot=IDENTITY):
        con.pos = np.asarray(pos, dtype=np.float32)
        con.rotmat = np.asarray(rot, dtype=np.float32)

    def _activate_ctx(arm):
        """Make sure mjc.actors matches the per-arm ctx before using it."""
        ctx = pln_ctx_lft if arm is lft else pln_ctx_rgt
        if mjc.actors != ctx.planning_mecbas:
            mjc.actors = list(ctx.planning_mecbas)
        return ctx

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
        ctx = _activate_ctx(arm)
        other = rgt if arm is lft else lft
        ctx.set_aux_mecbas(other, qs=other_arm_qs)
        ctx.set_aux_mecbas(lft.gripper, qs=(open_jaw / 2, open_jaw / 2))
        ctx.set_aux_mecbas(rgt.gripper, qs=(open_jaw / 2, open_jaw / 2))
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
            for qs in qs_list:
                if ctx.is_state_valid(qs):
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
    ctx_l = _activate_ctx(lft)
    ctx_l.set_aux_mecbas(rgt, qs=home_rgt)
    ctx_l.set_aux_mecbas(lft.gripper, qs=(open_jaw / 2, open_jaw / 2))
    ctx_l.set_aux_mecbas(rgt.gripper, qs=(open_jaw / 2, open_jaw / 2))
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
            (q for q in ql_pick if ctx_l.is_state_valid(q)), None)
        if q_pick is None:
            continue
        q_ho = next(
            (q for q in ql_ho if ctx_l.is_state_valid(q)), None)
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
        ctx = _activate_ctx(active)
        plnr = planner_lft if active is lft else planner_rgt
        other = rgt if active is lft else lft
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

    print("All plans OK. Animating.")

    # Reset to start for animation
    lft.fk(qs=home_lft)
    rgt.fk(qs=home_rgt)
    lft.gripper.set_jaw_width(open_jaw)
    rgt.gripper.set_jaw_width(open_jaw)
    set_obj(PICK_POS)

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

    # Each phase: (active_arm, 6-DOF path, frozen_other_qs, on_enter)
    phases = [
        (lft, pathA_q, home_rgt, None),
        (lft, pathB_q, home_rgt, attach_left),
        (lft, pathC_q, home_rgt, release_left_at_handover),
        (rgt, pathD_q, home_lft, None),
        (rgt, pathE_q, home_lft, attach_right),
        (rgt, pathF_q, home_lft, release_right_at_place),
    ]

    phase_idx = [0]
    cursor = [0]
    entered = [False]

    def tick(dt):
        if phase_idx[0] >= len(phases):
            return
        active, path, other_qs, on_enter = phases[phase_idx[0]]
        if not entered[0]:
            if on_enter is not None:
                on_enter()
            entered[0] = True
        if cursor[0] < len(path):
            q = path[cursor[0]]
            if active is lft:
                lft.fk(qs=q)
                rgt.fk(qs=other_qs)
            else:
                rgt.fk(qs=q)
                lft.fk(qs=other_qs)
            cursor[0] += 1
        else:
            phase_idx[0] += 1
            cursor[0] = 0
            entered[0] = False

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == "__main__":
    main()
