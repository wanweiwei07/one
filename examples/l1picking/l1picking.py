"""L1O6 left-arm pick of an upright cylinder standing on a table, using grasps
pre-planned + saved by o6planning.py.

Pipeline:
    load JSON grasps (O6, ``PRIMITIVE`` grasp, cylinder-LOCAL frame)
      -> transform onto the cylinder where it stands on the table
      -> keep the ones with a collision-free, reachable IK (hand clears the
         table; the cylinder itself is the target so hand-vs-cylinder is allowed)
      -> motion-plan the FIRST feasible grasp: home -> pre-grasp -> grasp -> lift
      -> animate.

Table: tabletop + 4 legs, origin at the bottom-centre, placed at x=0.3,y=0,z=0;
tabletop top at z=0.9, length(y)=1.2, width(x)=0.6. Cylinder (dia 0.05, h 0.3)
stands upright on the tabletop.

Keys:  F = step one frame   G = play/pause (continuous)
       R = replay           N = next candidate grasp (re-plans)
Run headless validation (no window):  ONE_HEADLESS=1
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc                              # noqa: E402
import one.scene.scene_object as osso                         # noqa: E402
import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.collider.mj_collider as ocm                        # noqa: E402
import one.motion.core.planning_context as omppc     # noqa: E402
import one.motion.probabilistic.rrt as ompr                   # noqa: E402
import one.motion.primitives.approach_depart as ompad            # noqa: E402
import one.robots.base.tcp as orbt                            # noqa: E402
import one.robots.humanoids.linx.l1.l1 as l1                  # noqa: E402
from one.grasp.serialize import load_grasps, transform_grasps  # noqa: E402

GRASPS_JSON = os.path.join(_THIS, "o6_cylinder_grasps.json")
CHAIN = 'left_arm'      # left arm only (6-DOF, analytic S456X12); waist frozen
PRIMITIVE = 'tripod'     # grasp primitive -- MUST match the one o6planning saved
                        # the JSON with (drives the jaw tcp AND the closing)

# Table: origin at bottom-centre, at world (0.3, 0, 0). Tabletop top z=0.9.
TABLE_ORIGIN = np.array([0.3, 0.0, 0.0], dtype=np.float32)
TABLE_LEN_Y = 1.2            # length along y
TABLE_WID_X = 0.6           # width along x
TABLE_TOP_Z = 0.9
TOP_THICK = 0.04
LEG = 0.05                   # square leg cross-section
TABLE_RGB = (0.55, 0.42, 0.30)

# Cylinder mesh standing on the tabletop -- the SAME mesh o6planning planned the
# grasps on, so the saved local grasps map straight on via cyl.wd_tf. Its base is
# at z=0 in its own frame, so cyl.pos places the base on the tabletop.
CYL_STL = os.path.join(_THIS, "cylinder.stl")
CYL_POS = np.array([0.22, 0.3, TABLE_TOP_Z], dtype=np.float32)

UP = np.array([0.0, 0.0, 0.15], dtype=np.float32)   # lift after grasp


# ----------------------------------------------------------------------------
def build_table():
    """Tabletop + 4 legs as separate axis-aligned boxes (origin = bottom centre
    at TABLE_ORIGIN). Returns the list of parts."""
    ox, oy, _ = TABLE_ORIGIN
    top_cz = TABLE_TOP_Z - TOP_THICK / 2
    parts = [ossop.box(xyz_lengths=(TABLE_WID_X, TABLE_LEN_Y, TOP_THICK),
                       pos=(ox, oy, top_cz), rgb=TABLE_RGB,
                       collision_type=ouc.CollisionType.AABB)]
    leg_h = TABLE_TOP_Z - TOP_THICK
    inset_x = TABLE_WID_X / 2 - LEG / 2
    inset_y = TABLE_LEN_Y / 2 - LEG / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(ossop.box(
                xyz_lengths=(LEG, LEG, leg_h),
                pos=(ox + sx * inset_x, oy + sy * inset_y, leg_h / 2),
                rgb=TABLE_RGB, collision_type=ouc.CollisionType.AABB))
    return parts


def build_scene():
    robot = l1.L1O6()
    table = build_table()
    # File-backed mesh, so the MuJoCo collider can use MESH collision directly
    # (a procedural cylinder has no file and previously needed a CAPSULE fallback).
    cyl = osso.SceneObject.from_file(
        CYL_STL, collision_type=ouc.CollisionType.MESH,
        is_floating=True, rgb=(0.6, 0.7, 0.5))
    cyl.pos = CYL_POS.copy()
    ground = ossop.plane(pos=(0, 0, 0.0))
    return robot, table, cyl, ground


def make_collider(robot, table, cyl, ground):
    """Collider for motion planning: robot vs table + cylinder + ground + self.
    The cylinder IS an obstacle so the free-space approach (home -> pre-grasp)
    and the pre-grasp pose avoid it; the grasp/lift keyframes are checked with
    ``collision_free=False`` (and pre->grasp is a straight interpolation), so the
    intended hand-vs-cylinder contact at the grasp is not flagged.

    The scene cylinder is ``is_floating=True`` (it gets grasped/lifted); a free body
    is not pinned at its pose in the collider and would float to the origin and
    spuriously collide, so the obstacle is a FIXED clone at the cylinder's pose."""
    cyl_obstacle = cyl.clone()
    cyl_obstacle.is_floating = False
    # pinned clone used purely as a static obstacle here: declare it STATIC so it
    # does not spuriously contact the table/ground it rests on (STATIC-vs-STATIC
    # is off). The role no longer follows is_floating, so it is declared explicitly.
    cyl_obstacle.collision_group = ouc.CollisionGroup.STATIC
    mjc = ocm.MJCollider()
    for e in (robot, *table, cyl_obstacle, ground):
        mjc.append(e)
    mjc.actors = [robot]
    mjc.compile(margin=0.0, auto_acm=True)
    return mjc


def chain_planning_context(robot, mjc, chain_name):
    """PlanningContext over the full qs with every joint NOT on ``chain_name``
    frozen at home -> the planner only explores that chain."""
    c = robot._compiled
    chain = robot.chain(chain_name)
    lo = c.jlmt_low_by_idx.astype(np.float64).copy()
    hi = c.jlmt_high_by_idx.astype(np.float64).copy()
    home = robot.qs.astype(np.float64).copy()
    free = np.zeros(c.n_jnts, dtype=bool)
    free[chain.active_jnt_ids] = True
    lo[~free] = home[~free]
    hi[~free] = home[~free]
    return omppc.PlanningContext(collider=mjc, joint_limits=(lo, hi))


def load_world_grasps(cyl):
    """Load the cylinder-local grasps and map them onto the cylinder's world
    pose. Returns [(pose_world, pre_world, jaw_width, score), ...]."""
    return transform_grasps(load_grasps(GRASPS_JSON), cyl.wd_tf)


def build_motion(robot, ctx, planner, jaw, grasp):
    """Pick motion for one grasp = (pose, pre_pose, jw, score):
    home -> pre-grasp (planned) -> grasp (approach) -> lift. Returns
    (traj, grasp_idx, amount) or None if a keyframe is unreachable/colliding."""
    pose, pre_pose, jw, _sc = grasp
    rot = pose[:3, :3]
    g = pose[:3, 3]
    # per-grasp power-center tcp on the mounted hand (the calibrated grasp
    # centre for this closure) -- IK puts the pads where they will close.
    # Use the FULL grasp-center loc_tf (position AND orientation): jaw's
    # grasp frame is rotated relative to the hand base, and grip_at applies
    # that rotation. Feeding IK a position-only (identity-rotation) tcp would
    # reach the right point with the wrong wrist twist -> hand hits the object.
    center_tcp = orbt.TCP(robot.left_hand.runtime_root_lnk,
                          jaw.eval_grasp_tcp(jw).loc_tf)
    home = robot.qs.astype(np.float64).copy()
    # bind the planning session + this grasp's center tcp once (the tcp depends
    # on the jaw width, so it is per-grasp); both moves then pass only what varies.
    adp = ompad.ADPlanner(robot, ctx, planner, chain=CHAIN, tcp=center_tcp)
    # RRT home -> pre-grasp, then a straight CARTESIAN descent into the grasp
    # (the pre pose is loaded with the grasp, so pass it explicitly). The hand
    # comes straight down the approach axis instead of bowing sideways the way a
    # joint-space line would. check_descent=False: the target cylinder is a
    # collision obstacle here, so the intended hand-vs-cylinder contact at the
    # grasp must NOT gate the descent (mirrors the old collision_free=False).
    approach = adp.gen_approach(
        g, rot, start_qs=home, pre_pos=pre_pose[:3, 3],
        pre_rotmat=pre_pose[:3, :3], granularity=0.01, use_rrt=True,
        check_descent=False, max_iters=4000)
    if approach is None:
        return None
    grasp_idx = len(approach) - 1               # hand closes here (at the grasp)
    grasp_q = approach.jv_list[-1]
    # straight cartesian lift (carrying); contact still intended -> ungated.
    lift = adp.gen_depart(
        g, rot, start_qs=grasp_q, depart_direction=UP,
        depart_distance=float(np.linalg.norm(UP)),
        granularity=0.01, check_retreat=False)
    if lift is None:
        return None
    traj = (approach + lift).jv_list
    amount = float(jaw._amount_for(jw))         # jw -> power closure amount
    return traj, grasp_idx, amount


def first_reachable(robot, ctx, planner, jaw, grasps, start=0):
    """Build the motion for the first feasible grasp at/after ``start`` (wrap).
    Returns (grasp_index, motion) or (None, None)."""
    for k in range(len(grasps)):
        idx = (start + k) % len(grasps)
        motion = build_motion(robot, ctx, planner, jaw, grasps[idx])
        if motion is not None:
            return idx, motion
    return None, None


# ----------------------------------------------------------------------------
def main():
    headless = bool(os.environ.get("ONE_HEADLESS"))
    if headless:
        np.random.seed(0)
    robot, table, cyl, ground = build_scene()
    mjc = make_collider(robot, table, cyl, ground)   # cylinder is an obstacle
                                                     # until the grasp itself
    ctx = chain_planning_context(robot, mjc, CHAIN)
    planner = ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=np.pi / 36,
                                     goal_bias=0.3)
    jaw = robot.left_hand.spawn_jaw(PRIMITIVE)
    grasps = load_world_grasps(cyl)
    print(f"loaded {len(grasps)} grasps from {os.path.basename(GRASPS_JSON)}")

    gi, motion = first_reachable(robot, ctx, planner, jaw, grasps)
    if motion is None:
        raise RuntimeError("no loaded grasp is reachable + collision-free")
    traj, grasp_idx, _amount = motion
    print(f"first feasible grasp {gi} (score {grasps[gi][3]:.2f}): "
          f"{len(traj)} waypoints (grasp@{grasp_idx})")

    if headless:
        # count how many grasps survive the reachable + collision-free filter
        n_ok = sum(build_motion(robot, ctx, planner, jaw, g) is not None
                   for g in grasps[:20])
        print(f"headless OK: {n_ok}/20 sampled grasps feasible; "
              f"first feasible = grasp {gi}")
        return

    import builtins
    import pyglet.window.key as key
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(1.6, 1.0, 1.7), cam_lookat_pos=(0.35, 0.1, 0.95))
    builtins.base = base
    ossop.frame().attach_to(base.scene)
    for e in (robot, *table, cyl, ground):
        e.attach_to(base.scene)

    # # Draw every loaded grasp as translucent ghost jaws: GREEN at the grasp
    # # ``pose`` (closed to the grasp width), YELLOW at the ``pre`` approach pose
    # # (jaw open). All at alpha 0.3 so the picking animation stays readable.
    # jaw_open = float(jaw.jaw_range[1])
    # for pose, pre, jw, _sc in grasps:
    #     ghost_pose = robot.left_hand.spawn_jaw('power')
    #     ghost_pose.grip_at(pose[:3, 3], pose[:3, :3], jw)
    #     ghost_pose.rgb = (0.20, 0.85, 0.25)
    #     ghost_pose.alpha = 0.3
    #     ghost_pose.attach_to(base.scene)
    #     ghost_pre = robot.left_hand.spawn_jaw('power')
    #     ghost_pre.grip_at(pre[:3, 3], pre[:3, :3], jaw_open)
    #     ghost_pre.rgb = (0.95, 0.85, 0.15)
    #     ghost_pre.alpha = 0.3
    #     ghost_pre.attach_to(base.scene)

    cyl_home = (cyl.pos.copy(), cyl.rotmat.copy())
    print("F: step frame   G: play/pause   R: replay   N: next grasp")

    state = {"gi": gi, "motion": motion, "i": 0, "held": False, "playing": False}

    def reset_play():
        if state["held"]:
            robot.left_hand.release(cyl)
            state["held"] = False
        robot.left_hand.open_hand()
        cyl.pos, cyl.rotmat = cyl_home[0].copy(), cyl_home[1].copy()
        robot.fk(qs=state["motion"][0][0])
        state["i"] = 0
        base.scene.dirty = True

    def select_next():
        reset_play()
        gi2, motion2 = first_reachable(robot, ctx, planner, jaw, grasps,
                                       start=state["gi"] + 1)
        if motion2 is None:
            print("no other feasible grasp")
            return
        state["gi"], state["motion"] = gi2, motion2
        print(f"grasp {gi2} (score {grasps[gi2][3]:.2f}): "
              f"{len(motion2[0])} waypoints")
        reset_play()

    def step_one():
        traj_i, grasp_idx_i, amount = state["motion"]
        i = state["i"]
        if i >= len(traj_i):
            state["playing"] = False
            return
        robot.fk(qs=traj_i[i])
        if i == grasp_idx_i and not state["held"]:
            robot.left_hand.grasp(cyl, primitive=PRIMITIVE, amount=amount)
            state["held"] = True
        state["i"] += 1
        base.scene.dirty = True

    reset_play()

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.R):
            reset_play()
            return
        if base.input_manager.is_key_pressed_edge(key.N):
            select_next()
            return
        if base.input_manager.is_key_pressed_edge(key.G):
            state["playing"] = not state["playing"]
        if base.input_manager.is_key_pressed_edge(key.F):
            state["playing"] = False
            step_one()
        if state["playing"]:
            step_one()

    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
