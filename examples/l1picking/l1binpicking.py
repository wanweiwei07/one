"""Bin-picking demo: cylinder.stl parts settle into the LEFT bin (MuJoCo), then
the L1O6 left arm picks one, carries it over the RIGHT bin, and releases it so it
free-falls into that bin (MuJoCo again). Repeat for the next part.

Two decoupled systems share the SAME SceneObjects:
  * physics scene (cylinders + static bins/table/ground, NO robot)  -> MJEnv
    drives the settle and the post-release drop.
  * planning collider (robot + statics + the OTHER cylinders)       -> kinematic
    motion planning; the robot is moved purely by fk, never by dynamics (so the
    un-actuated arm can't droop under gravity).

Bin layout is read live from BIN_* below (left = pick, right = place).

Keys:  F = step one frame   G = play/pause   R = replay this pick
       N = next part (re-plan)
Headless (settle + plan first pick, no window):  ONE_HEADLESS=1
"""
import os
import sys
import builtins
import colorsys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
for p in (_PROJECT_ROOT, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import one.utils.constant as ouc                               # noqa: E402
import one.utils.math as oum                                   # noqa: E402
import one.scene.scene as oss                                  # noqa: E402
import one.scene.scene_object as osso                          # noqa: E402
import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.collider.mj_collider as ocm                         # noqa: E402
import one.physics.mj_env as opme                              # noqa: E402
import one.motion.probabilistic.rrt as ompr                    # noqa: E402
import one.motion.primitives.approach_depart as ompad             # noqa: E402
import one.robots.humanoids.linx.l1.l1 as l1                   # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402
from one.grasp.serialize import load_grasps, transform_grasps  # noqa: E402

from l1picking import (build_table, TABLE_TOP_Z, CHAIN,        # noqa: E402
                       chain_planning_context)

CYL_STL = os.path.join(_THIS, "cylinder.stl")
GRASPS_JSON = os.path.join(_THIS, "o6_cyl_stl_grasps.json")

# bin geometry (left = pick, right = place)
BIN_BOTTOM = (0.35, 0.25, 0.03)   # x, y, thickness
# shallow tray: the O6 hand is too bulky to dip into a deep bin -- with walls
# taller than ~3cm NO grasp clears the wall (parts settle ~8cm below a 10cm rim).
WALL_H = 0.03
WALL_T = 0.01
BIN_CX = 0.30
LEFT_CY = 0.30
RIGHT_CY = 0.025
BIN_RGB = (0.32, 0.34, 0.40)

N_CYL = 20
UP = np.array([0.0, 0.0, 0.12], dtype=np.float32)             # lift after grasp
# release point: above the RIGHT bin, clear of its walls
PLACE_POS = np.array([BIN_CX, RIGHT_CY, TABLE_TOP_Z + WALL_H + 0.16],
                     dtype=np.float32)
# the dropped part free-falls, so its yaw at release is irrelevant -- we try
# these place orientations (grasp rot spun about world z) and keep one whose IK
# has a NORMAL elbow (joint 3 near 0, not the q3~180 flipped branch).
PLACE_YAWS = np.radians(np.arange(0, 360, 30))
ELBOW_IDX = 2                                    # active joint 3 (the elbow)
ELBOW_MAX = np.deg2rad(90.0)                      # |q3| below this == normal


def build_bin(cx, cy, z_top):
    bx, by, bt = BIN_BOTTOM
    parts = [ossop.box(pos=(cx, cy, z_top + bt / 2),
                       xyz_lengths=(bx, by, bt), rgb=BIN_RGB,
                       collision_type=ouc.CollisionType.AABB, is_floating=False)]
    wz = z_top + bt + WALL_H / 2
    for sx in (-1, 1):
        parts.append(ossop.box(
            pos=(cx + sx * (bx / 2 + WALL_T / 2), cy, wz),
            xyz_lengths=(WALL_T, by, WALL_H), rgb=BIN_RGB,
            collision_type=ouc.CollisionType.AABB, is_floating=False))
    for sy in (-1, 1):
        parts.append(ossop.box(
            pos=(cx, cy + sy * (by / 2 + WALL_T / 2), wz),
            xyz_lengths=(bx + 2 * WALL_T, WALL_T, WALL_H), rgb=BIN_RGB,
            collision_type=ouc.CollisionType.AABB, is_floating=False))
    return parts


def scatter_cylinders(cx, cy, z_floor, n, seed=0):
    rng = np.random.default_rng(seed)
    sp = 0.085
    nx, ny = 3, 3
    xs = (np.arange(nx) - (nx - 1) / 2) * sp
    ys = (np.arange(ny) - (ny - 1) / 2) * sp
    per_layer = nx * ny
    cyls = []
    for i in range(n):
        layer, k = divmod(i, per_layer)
        px = cx + xs[k % nx] + rng.uniform(-0.015, 0.015)
        py = cy + ys[k // nx] + rng.uniform(-0.015, 0.015)
        pz = z_floor + 0.05 + layer * 0.10
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rgb = colorsys.hsv_to_rgb(i / n, 0.55, 0.92)
        # CAPSULE collision: MuJoCo can't settle MESH/native-CYLINDER stably for
        # many small parts (jitter/tunnelling); capsule is the stable proxy. The
        # STL stays the visual.
        c = osso.SceneObject.from_file(
            CYL_STL, collision_type=ouc.CollisionType.CAPSULE, is_floating=True,
            rgb=rgb)
        c.set_pos_rotmat(pos=np.array([px, py, pz], dtype=np.float32),
                         rotmat=oum.rotmat_from_axangle(axis,
                                                        rng.uniform(0, np.pi)))
        cyls.append(c)
    return cyls


# ---------------------------------------------------------------------------
def build_world():
    """robot + statics + scattered cylinders, all rendered in base.scene; a
    separate physics Scene holds statics + cylinders (no robot)."""
    robot = l1.L1O6()
    statics = ([ossop.plane(pos=(0, 0, 0.0))] + list(build_table())
               + build_bin(BIN_CX, LEFT_CY, TABLE_TOP_Z)
               + build_bin(BIN_CX, RIGHT_CY, TABLE_TOP_Z))
    bin_floor = TABLE_TOP_Z + BIN_BOTTOM[2]
    cyls = scatter_cylinders(BIN_CX, LEFT_CY, bin_floor, N_CYL)
    phys = oss.Scene()
    for e in statics + cyls:
        phys.add(e)
    return robot, statics, cyls, phys


def make_pick_collider(robot, statics, cyls, target):
    """Motion-planning collider: robot vs statics vs every cylinder EXCEPT the
    one being grasped (that one is carried, not an obstacle).

    The obstacle cylinders are cloned and declared STATIC at their current settled
    pose. is_collided() counts global contacts, and a pile of ACTIVE cylinders
    resting on each other and the bin produces dozens of permanent ACTIVE-ACTIVE /
    ACTIVE-STATIC contacts that would mark every robot config as collided.
    STATIC-vs-STATIC pairs are filtered by the collision matrix, so the fixed
    clones still block the robot (ACTIVE-vs-STATIC is checked) without the pile
    noise. The live cylinders stay in the physics scene untouched."""
    mjc = ocm.MJCollider()
    obstacles = []
    for c in cyls:
        if c is target:
            continue
        fixed = c.clone()
        fixed.is_floating = False
        # pinned clone used as a static obstacle: declare STATIC (the role no
        # longer follows is_floating) so it does not contact its neighbours/ground.
        fixed.collision_group = ouc.CollisionGroup.STATIC
        obstacles.append(fixed)
    for e in [robot] + statics + obstacles:
        mjc.append(e)
    mjc.actors = [robot]
    mjc.compile(margin=0.0, auto_acm=True)
    return mjc


def normal_elbow_ik(robot, ctx, pos, rot, tcp, ref):
    """Nearest-to-``ref`` IK config at (pos, rot) that has a NORMAL elbow
    (|joint3| < ELBOW_MAX, i.e. not the q3~180 flipped branch), stays in-limit
    and is collision-free. None if no such solution -- the caller then skips
    this grasp/orientation rather than execute a reversed elbow."""
    chain = robot.chain(CHAIN)
    ref_active = chain.extract_active_qs(ref)
    best = None
    for s in robot.ik(pos, rot, chain=CHAIN, tcp=tcp,
                      ref_qs=ref_active, max_solutions=8):
        a = chain.extract_active_qs(s)
        if abs(a[ELBOW_IDX]) >= ELBOW_MAX:            # flipped elbow -> skip
            continue
        if not ctx.is_state_valid(s.astype(np.float64)):
            continue
        d = float(np.linalg.norm(a - ref_active))
        if best is None or d < best[0]:
            best = (d, s.astype(np.float64))
    return None if best is None else best[1]


def place_config(robot, ctx, pos, base_rot, tcp, ref):
    """Release config over the right bin. The dropped part free-falls, so its
    yaw is irrelevant: try several orientations (base_rot spun about world z) and
    return the normal-elbow config nearest to ``ref``, plus the orientation it
    used -- or None. Nearest-to-ref keeps the lift->place motion small AND
    prefers the carried orientation."""
    chain = robot.chain(CHAIN)
    ref_active = chain.extract_active_qs(ref)
    best = None
    for yaw in PLACE_YAWS:
        rot = oum.rotmat_from_axangle(ouc.StandardAxis.Z, yaw) @ base_rot
        q = normal_elbow_ik(robot, ctx, pos, rot, tcp, ref)
        if q is None:
            continue
        d = float(np.linalg.norm(chain.extract_active_qs(q) - ref_active))
        if best is None or d < best[0]:
            best = (d, q, rot)
    return None if best is None else (best[1], best[2])


def build_pick_place(robot, ctx, planner, jaw, grasp):
    """home -> pre -> grasp -> lift -> place(over right bin) -> (release), for a
    world-frame :class:`Grasp`.
    Returns (traj, grasp_idx, amount) or None if any keyframe is infeasible.

    The target cylinder is excluded from the collider (make_pick_collider), so
    every config is collision-CHECKED: contacting the absent target is free, but
    the hand may NOT pierce the bin walls or neighbour parts. The pre->grasp
    descent and grasp->lift retreat are straight CARTESIAN moves (ompad.gen_*,
    check_*=True so ctx gates them) so the hand goes straight along the approach
    axis instead of bowing into a wall."""
    wpose, wpre = grasp.pose, grasp.pre_pose
    jw = grasp.provenance["jaw_width"]
    rot, g = wpose[:3, :3], wpose[:3, 3]
    tcp = grasp.make_tcp(robot.left_hand)
    home = robot.qs.astype(np.float64).copy()
    chain = robot.chain(CHAIN)
    # NORMAL-elbow gate (joint3 near 0, not the q3~180 flipped branch): the grasp
    # orientation is fixed by the antipodal contact, so we can't spin it like
    # place; the cartesian descent/lift inherit this elbow branch via seeded IK.
    normal_elbow = lambda q: abs(chain.extract_active_qs(q)[ELBOW_IDX]) < ELBOW_MAX
    adp = ompad.ADPlanner(robot, ctx, planner, chain=CHAIN, tcp=tcp)
    # RRT home -> pre-grasp, then a straight CARTESIAN descent into the grasp;
    # ctx gates the descent (target excluded -> contacting it is free, but bin
    # walls / neighbours are not), and ik_accept enforces the normal elbow.
    approach = adp.gen_approach(
        g, rot, start_qs=home, pre_pos=wpre[:3, 3], pre_rotmat=wpre[:3, :3],
        granularity=0.01, use_rrt=True, check_descent=True, max_iters=4000,
        ik_accept=normal_elbow)
    if approach is None:                        # no normal-elbow pre / wall clip
        return None
    grasp_idx = len(approach) - 1               # hand closes here (at grasp_q)
    grasp_q = approach.jv_list[-1]
    # straight cartesian retreat back up (still collision-gated)
    lift = adp.gen_depart(
        g, rot, start_qs=grasp_q, depart_direction=UP,
        depart_distance=float(np.linalg.norm(UP)), granularity=0.01,
        check_retreat=True)
    if lift is None:
        return None
    placed = place_config(robot, ctx, PLACE_POS, rot, tcp, lift.jv_list[-1])
    if placed is None:                          # no normal-elbow place -> skip
        return None
    place, _ = placed
    lift_to_place = planner.solve(lift.jv_list[-1], place, max_iters=4000)
    if not lift_to_place:
        return None
    # pre already present in approach; grasp_q seam between approach & lift.
    traj = approach.jv_list + lift.jv_list[1:] + list(lift_to_place)
    amount = float(jaw._amount_for(jw))
    return traj, grasp_idx, amount


def plan_next_pick(robot, statics, cyls, grasps_local, picked, jaw):
    """First feasible pick over the un-picked cylinders (topmost first)."""
    order = sorted((i for i in range(len(cyls)) if i not in picked),
                   key=lambda i: -float(cyls[i].pos[2]))
    for ci in order:
        target = cyls[ci]
        mjc = make_pick_collider(robot, statics, cyls, target)
        ctx = chain_planning_context(robot, mjc, CHAIN)
        planner = ompr.RRTConnectPlanner(pln_ctx=ctx,
                                         extend_step_size=np.pi / 36,
                                         goal_bias=0.3)
        for grasp in transform_grasps(grasps_local, target.wd_tf):
            m = build_pick_place(robot, ctx, planner, jaw, grasp)
            if m is not None:
                return ci, m
    return None, None


# ---------------------------------------------------------------------------
def main():
    headless = bool(os.environ.get("ONE_HEADLESS"))
    base = ovw.World(cam_pos=(1.6, 0.8, 1.6), cam_lookat_pos=(0.30, 0.16, 0.95))
    builtins.base = base
    robot, statics, cyls, phys = build_world()
    ossop.frame().attach_to(base.scene)
    for e in [robot] + statics + cyls:
        e.attach_to(base.scene)

    mjenv = opme.MJEnv(scene=phys)
    print(f"settling {N_CYL} cylinders ...")
    for _ in range(900):
        mjenv.step(0.01)

    grasps_local = load_grasps(GRASPS_JSON)
    jaw = robot.left_hand.as_jaw('pinch')
    picked = set()
    gi, motion = plan_next_pick(robot, statics, cyls, grasps_local, picked, jaw)
    if motion is None:
        raise RuntimeError("no feasible pick found on the settled pile")
    print(f"first pick: cylinder {gi}, {len(motion[0])} waypoints "
          f"(grasp@{motion[1]})")

    if headless:
        return

    import pyglet.window.key as key

    st = {"gi": gi, "motion": motion, "i": 0, "held": False,
          "playing": False, "dropping": False, "drop_steps": 0}

    def reset_play():
        if st["held"]:
            robot.left_hand.detach(cyls[st["gi"]])   # open_hand below reopens
            st["held"] = False
        robot.left_hand.open_hand()
        robot.fk(qs=st["motion"][0][0])
        st["i"] = 0
        st["dropping"] = False
        st["drop_steps"] = 0
        base.scene.dirty = True

    def step_traj():
        traj, grasp_idx, amount = st["motion"]
        i = st["i"]
        if i >= len(traj):                       # trajectory done -> release+drop
            if st["held"]:
                robot.left_hand.open_hand()          # reopen, then drop
                robot.left_hand.detach(cyls[st["gi"]])
                st["held"] = False
                mjenv.sync.push_all_sobj_qpos()  # sync carried pose into mujoco
                st["dropping"] = True
            return
        robot.fk(qs=traj[i])
        if i == grasp_idx and not st["held"]:
            robot.left_hand.grip('pinch', amount)
            robot.left_hand.attach(cyls[st["gi"]])
            st["held"] = True
        st["i"] += 1
        base.scene.dirty = True

    def step_drop():
        mjenv.step(0.02)
        st["drop_steps"] += 1
        if st["drop_steps"] >= 120:              # ~2.4 s of fall/settle
            st["dropping"] = False
            st["playing"] = False
        base.scene.dirty = True

    def step_one():
        if st["dropping"]:
            step_drop()
        else:
            step_traj()

    def select_next():
        if st["gi"] not in st["pickedset"]:
            st["pickedset"].add(st["gi"])
        gi2, m2 = plan_next_pick(robot, statics, cyls, grasps_local,
                                 st["pickedset"], jaw)
        if m2 is None:
            print("no further feasible pick")
            return
        st["gi"], st["motion"] = gi2, m2
        print(f"next pick: cylinder {gi2}, {len(m2[0])} waypoints")
        reset_play()

    st["pickedset"] = picked
    reset_play()

    def tick(dt):
        im = base.input_manager
        if im.is_key_pressed_edge(key.R):
            reset_play(); return
        if im.is_key_pressed_edge(key.N):
            select_next(); return
        if im.is_key_pressed_edge(key.G):
            st["playing"] = not st["playing"]
        if im.is_key_pressed_edge(key.F):
            st["playing"] = False
            step_one()
        if st["playing"]:
            step_one()

    print("F: step   G: play/pause   R: replay   N: next part")
    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
