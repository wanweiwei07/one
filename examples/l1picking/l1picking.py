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
import one.motion.probabilistic.planning_context as omppc     # noqa: E402
import one.motion.probabilistic.rrt as ompr                   # noqa: E402
import one.robots.base.tcp as orbt                            # noqa: E402
import one.robots.humanoids.linx.l1.l1 as l1                  # noqa: E402
from one.grasp.serialize import load_grasps                   # noqa: E402

GRASPS_JSON = os.path.join(_THIS, "o6_cylinder_grasps.json")
CHAIN = 'left_arm'      # left arm only (6-DOF, analytic S456X12); waist frozen
PRIMITIVE = 'power'     # grasp primitive -- MUST match the one o6planning saved
                        # the JSON with (drives the jaw tcp AND the closing)

# Table: origin at bottom-centre, at world (0.3, 0, 0). Tabletop top z=0.9.
TABLE_ORIGIN = np.array([0.3, 0.0, 0.0], dtype=np.float32)
TABLE_LEN_Y = 1.2            # length along y
TABLE_WID_X = 0.6           # width along x
TABLE_TOP_Z = 0.9
TOP_THICK = 0.04
LEG = 0.05                   # square leg cross-section
TABLE_RGB = (0.55, 0.42, 0.30)

# Cylinder (dia 0.05, h 0.3) standing on the tabletop, at a reachable spot.
CYL_RADIUS = 0.025
CYL_HEIGHT = 0.30
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
    # Capsule collision (not mesh): the MuJoCo collider needs a native primitive
    # or a file-backed mesh, and a procedural cylinder mesh has no file. A
    # capsule is the natural fit for a cylinder (radius matches; rounded ends are
    # slightly conservative) and the cylinder is only an obstacle here -- grasps
    # are pre-loaded, not re-planned against this shape.
    cyl = ossop.cylinder(
        spos=(0.0, 0.0, 0), epos=(0.0, 0.0, CYL_HEIGHT),
        radius=CYL_RADIUS, segments=24,
        collision_type=ouc.CollisionType.CAPSULE,
        is_free=True, rgb=(0.6, 0.7, 0.5))
    cyl.pos = CYL_POS.copy()
    ground = ossop.plane(pos=(0, 0, 0.0))
    return robot, table, cyl, ground


def make_collider(robot, table, cyl, ground):
    """Collider for motion planning: robot vs table + cylinder + ground + self.
    The cylinder IS an obstacle so the free-space approach (home -> pre-grasp)
    and the pre-grasp pose avoid it; the grasp/lift keyframes are checked with
    ``collision_free=False`` (and pre->grasp is a straight interpolation), so the
    intended hand-vs-cylinder contact at the grasp is not flagged.

    The scene cylinder is ``is_free=True`` (it gets grasped/lifted); a free body
    is not pinned at its pose in the collider and would float to the origin and
    spuriously collide, so the obstacle is a FIXED clone at the cylinder's pose."""
    cyl_obstacle = cyl.clone()
    cyl_obstacle.is_free = False
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
    local = load_grasps(GRASPS_JSON)
    tf = cyl.wd_tf
    return [(tf @ pose, tf @ pre, jw, sc) for pose, pre, jw, sc in local]


def ik_config(robot, ctx, pos, rotmat, tcp, collision_free=True, ref=None):
    """IK for ``tcp`` at (pos, rotmat); among the (optionally collision-free)
    solutions return the one closest to ``ref`` in joint space. Returns full
    qs (n_jnts,) or None."""
    chain = robot.chain(CHAIN)
    if ref is None:
        ref = robot.qs
    ref_active = chain.extract_active_qs(ref)
    sols = robot.ik(pos, rotmat, chain=CHAIN, tcp=tcp,
                    ref_qs=ref_active, max_solutions=8)
    best, best_d = None, None
    for s in sols:
        if collision_free and not ctx.is_state_valid(s.astype(np.float64)):
            continue
        d = float(np.linalg.norm(chain.extract_active_qs(s) - ref_active))
        if best_d is None or d < best_d:
            best, best_d = s.astype(np.float64), d
    return best


def jtraj(q0, q1, step=np.deg2rad(3.0)):
    n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / step)))
    return [q0 + (q1 - q0) * t for t in np.linspace(0, 1, n)]


def plan_segment(planner, q0, q1):
    path = planner.solve(q0, q1, max_iters=4000)
    return path if path else jtraj(q0, q1)   # fall back to interpolation


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
    pre = ik_config(robot, ctx, pre_pose[:3, 3], pre_pose[:3, :3], center_tcp)
    grasp_q = ik_config(robot, ctx, g, rot, center_tcp,
                        collision_free=False, ref=pre)
    lift = ik_config(robot, ctx, g + UP, rot, center_tcp,
                     collision_free=False, ref=grasp_q)
    if any(x is None for x in (pre, grasp_q, lift)):
        return None
    traj = []
    traj += plan_segment(planner, home, pre)   # planned free-space approach
    traj += jtraj(pre, grasp_q)                 # straight approach to the grasp
    grasp_idx = len(traj) - 1
    traj += jtraj(grasp_q, lift)                # lift (carrying)
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
