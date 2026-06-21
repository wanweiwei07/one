"""RS007L pick-and-place of a bunny between two FIXED poses, composed entirely
from the existing building blocks -- NO pick_place planner yet, just:

    GraspReasoner.reason_common_gids([pick_pose, place_pose])   -> a grasp that
        is reachable + collision-free at BOTH the pick and the place pose.
    ADPlanner.gen_approach / gen_depart                          -> RRT to a
        pre-grasp then a straight cartesian line into the grasp, and the mirror
        retreat.
    MotionData (+)                                               -> compose the
        segments into one trajectory.

So the whole motion is:
    gen_approach(pick) + grasp + gen_depart(lift)
        + gen_approach(place) [RRT transfer + descent] + release + gen_depart(retreat)

This is the composition a future one/manipulation/pick_place.py would wrap once
its shape (fixed vs free-yaw place, single vs dual arm) settles. The bunny is the
MANIPULATED object, so it is excluded from the planning collider (like
l1binpicking) -- antipodal already guarantees its grasp clearance, and the held
bunny is shown carried (mounted on the gripper) rather than collision-checked
mid-transfer.

Run:        py -3.12 examples/test_rs007l_pickplace_bunny.py
Headless:   ONE_HEADLESS=1 py -3.12 examples/test_rs007l_pickplace_bunny.py
"""
import os

import numpy as np

import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
import one.collider.mj_collider as ocm
from one.grasp.antipodal import antipodal
from one import (oum, ouc, ossop, osso, khi_rs007l, or_2fg7,  # noqa: F401
                 omppc, ompr, ogr, ompad, MotionData)

HEADLESS = bool(os.environ.get("ONE_HEADLESS"))
BUNNY_STL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bunny.stl")
LIFT = 0.15                      # straight up after grasp / before release (m)
np.random.seed(7)               # antipodal samples randomly; seed for repeatability


# ----------------------------------------------------------------------------
# Scene (no viewer needed to plan -- the World is built later, only for the GUI)
robot = khi_rs007l.RS007L()
gripper = or_2fg7.OR2FG7()
robot.mount(gripper, robot.runtime_lnks[-1], update=True)
tcp = gripper.tcp('grasp_center')
OPEN = float(gripper.jaw_range[1])

bunny = osso.SceneObject.from_file(BUNNY_STL,
                                   collision_type=ouc.CollisionType.MESH,
                                   is_floating=True)   # manipulated -> must be mountable
ground = ossop.plane(pos=(0, 0, 0.01))

def jaw_qs(width):
    return (width / 2.0, width / 2.0)


# ----------------------------------------------------------------------------
# Pick and place poses: a stable rest pose of the bunny, placed at two spots the
# arm can reach (the same pair the shared-grasp demo uses, known to share grasps).
hull = ogf.convex_hull(bunny.collisions[0].geom)
facets = ogs.segment_surface(hull)
stable = ogp.compute_stable_poses(hull.vs, hull.fs, facets, com=None,
                                  stable_thresh=10.0)
assert stable, "no stable pose for bunny"
pos0, rot0 = stable[0][0], stable[0][1]
tf_pick = oum.tf_from_pos_rotmat(pos0 + np.array([-0.5, 0.0, 0.0], np.float32),
                                 rot0)
rot_place = oum.rotmat_from_euler(0, 0, np.deg2rad(45.0)) @ rot0
tf_place = oum.tf_from_pos_rotmat(pos0 + np.array([0.3, 0.3, 0.0], np.float32),
                                  rot_place)

# A partition wall standing between the pick and place, across the transfer
# corridor -- the robot must route the CARRIED bunny around it (only possible
# because the held bunny is collision-checked; see the carry collider below).
# mid = 0.5 * (tf_pick[:3, 3] + tf_place[:3, 3])
wall = ossop.box(pos=(0.0, 0.7, 0.10),
                 xyz_lengths=(0.05, 1.0, 0.3), name="wall",
                 collision_type=ouc.CollisionType.AABB)

# FREE-phase collider: robot + gripper + ground + wall, but NOT the bunny (the
# manipulated target -- antipodal guarantees its grasp clearance, so it is no
# obstacle while reaching to pick). The wall is a real obstacle in EVERY phase.
# auto_acm detects the arm's structural self-collisions ONCE; the carry collider
# reuses that set.
mjc_free = ocm.MJCollider()
for e in (robot, gripper, ground, wall):
    mjc_free.append(e)
mjc_free.actors = [robot]
mjc_free.compile(margin=0.0, auto_acm=True)
base_acm = list(mjc_free.acm)
ctx_free = omppc.PlanningContext(collider=mjc_free)
planner_free = ompr.RRTConnectPlanner(pln_ctx=ctx_free, goal_bias=0.3)


# ----------------------------------------------------------------------------
# Choose a grasp feasible at BOTH poses (resample grasps until one exists).
gid = None
grasps = []
for attempt in range(10):
    grasps = antipodal(gripper=gripper, target_sobj=bunny, density=0.006,
                       normal_tol_deg=25, roll_step_deg=30, max_grasps=120)
    if not grasps:
        continue
    reasoner = ogr.GraspReasoner(robot, ctx_free, grasps, tcp=tcp,
                                 gripper=gripper, max_solutions=1)
    common = reasoner.reason_common_gids([tf_pick, tf_place])
    if common:
        gid = sorted(common)[0]
        print(f"attempt {attempt}: {len(grasps)} grasps, "
              f"{len(common)} shared -> using grasp {gid}")
        break

if gid is None:
    raise SystemExit("no grasp shared between the pick and place poses")

pose, pre_pose, jw, score = grasps[gid]
pick_g = tf_pick @ pose
pick_pre = tf_pick @ pre_pose
place_g = tf_place @ pose
place_pre = tf_place @ pre_pose


# ----------------------------------------------------------------------------
# PICK (free phase): RRT home -> pre-grasp (gripper open) then straight down to
# the grasp, on the free collider (no bunny -> the descent into the target is
# environment-clear, check_descent=False, matching l1picking).
home = robot.qs.astype(np.float64).copy()
adp_free = ompad.ADPlanner(robot, ctx_free, planner_free, tcp=tcp)
mjc_free.set_mecba_qpos(gripper, jaw_qs(OPEN))
ctx_free.clear_cache()
pick = adp_free.gen_approach(
    pick_g[:3, 3], pick_g[:3, :3], start_qs=home,
    pre_pos=pick_pre[:3, 3], pre_rotmat=pick_pre[:3, :3],
    granularity=0.01, use_rrt=True, check_descent=False, ee_value=OPEN,
    max_iters=4000)
assert pick is not None, "pick approach infeasible"
q_grasp = pick.jv_list[-1]

# === GRASP: mount the bunny on the gripper (with the grasp offset), then build a
# CARRY collider where it rides the gripper link and IS collision-checked. The
# ACM is rule-based: reuse the arm's base ACM (no re-detection) + an explicit
# rule that the held bunny may touch the gripper's own links. ===
bunny.pos = tf_pick[:3, 3]
bunny.rotmat = tf_pick[:3, :3]
robot.fk(qs=q_grasp)
gripper.grasp(bunny, jaw_width=jw)                      # mount

mjc_carry = ocm.MJCollider()
for e in (robot, ground, wall):                         # bunny rides in via mount
    mjc_carry.append(e)
mjc_carry.actors = [robot]
mjc_carry.compile(margin=0.0, auto_acm=False,           # rule-based: no re-detect
                  extra_excludes=base_acm
                  + [(bunny, lnk) for lnk in gripper.runtime_lnks])
ctx_carry = omppc.PlanningContext(collider=mjc_carry)
planner_carry = ompr.RRTConnectPlanner(pln_ctx=ctx_carry, goal_bias=0.3)
adp_carry = ompad.ADPlanner(robot, ctx_carry, planner_carry, tcp=tcp)
mjc_carry.set_mecba_qpos(gripper, jaw_qs(jw))
ctx_carry.clear_cache()

# LIFT, then TRANSFER+PLACE on the CARRY collider -- the RRT now routes the HELD
# bunny (not just the gripper) around the wall.
lift = adp_carry.gen_depart(
    pick_g[:3, 3], pick_g[:3, :3], start_qs=q_grasp,
    depart_direction=(0, 0, 1), depart_distance=LIFT, granularity=0.01,
    ee_value=jw, check_retreat=False)
assert lift is not None, "lift infeasible"
place = adp_carry.gen_approach(
    place_g[:3, 3], place_g[:3, :3], start_qs=lift.jv_list[-1],
    pre_pos=place_pre[:3, 3], pre_rotmat=place_pre[:3, :3],
    granularity=0.01, use_rrt=True, check_descent=False, ee_value=jw,
    max_iters=8000)
assert place is not None, "place transfer infeasible (bunny cannot clear wall?)"
q_place = place.jv_list[-1]

# RETREAT straight up, released (gripper open). Short ungated cartesian move.
retreat = adp_carry.gen_depart(
    place_g[:3, 3], place_g[:3, :3], start_qs=q_place,
    depart_direction=(0, 0, 1), depart_distance=LIFT, granularity=0.01,
    ee_value=OPEN, check_retreat=False)
assert retreat is not None, "retreat infeasible"

gripper.release(bunny)        # drop the planning mount; the animation re-grasps

# Compose. grasp happens at the end of `pick`; release at the end of `place`.
before_release = pick + lift + place
grasp_idx = len(pick) - 1
release_idx = len(before_release) - 1
motion = before_release + retreat
print(f"motion: {len(motion)} waypoints "
      f"(grasp@{grasp_idx}, release@{release_idx})")

if HEADLESS:
    print("headless OK: pick-and-place with HELD-bunny collision built")
    raise SystemExit


# ----------------------------------------------------------------------------
# Viewer + animation. Starts PAUSED -- step one frame at a time with F to watch
# how the carried bunny jumps across the thin wall (tunneling between samples).
import one.viewer.world as ovw                                  # noqa: E402
import pyglet.window.key as key                                 # noqa: E402

base = ovw.World(cam_pos=(2.2, 2.2, 1.7), cam_lookat_pos=(0.1, 0.1, 0.6))
ossop.frame().attach_to(base.scene)
robot.attach_to(base.scene)
ground.attach_to(base.scene)
wall.rgb = (0.7, 0.4, 0.4)
wall.attach_to(base.scene)
ossop.frame(pos=tf_pick[:3, 3], rotmat=tf_pick[:3, :3]).attach_to(base.scene)
ossop.frame(pos=tf_place[:3, 3], rotmat=tf_place[:3, :3]).attach_to(base.scene)
bunny.pos = tf_pick[:3, 3]
bunny.rotmat = tf_pick[:3, :3]
bunny.attach_to(base.scene)

state = {"i": 0, "held": False, "playing": False}


def reset_play():
    if state["held"]:
        gripper.release(bunny)
        state["held"] = False
    bunny.pos = tf_pick[:3, 3]
    bunny.rotmat = tf_pick[:3, :3]
    gripper.set_jaw_width(OPEN)
    robot.fk(qs=motion.jv_list[0])
    state["i"] = 0
    state["playing"] = False
    base.scene.dirty = True


def step_one():
    i = state["i"]
    if i >= len(motion):
        state["playing"] = False
        return
    robot.fk(qs=motion.jv_list[i])
    ev = motion.ev_list[i]
    if ev is not None:
        gripper.set_jaw_width(ev)
    if i == grasp_idx and not state["held"]:
        gripper.grasp(bunny, jaw_width=jw)        # mount: bunny follows the tcp
        state["held"] = True
    if i == release_idx and state["held"]:
        gripper.release(bunny)
        state["held"] = False
        bunny.pos = tf_place[:3, 3]
        bunny.rotmat = tf_place[:3, :3]
    phase = ("pick" if i < grasp_idx else
             "carry" if i < release_idx else "place/retreat")
    print(f"frame {i}/{len(motion) - 1}  [{phase}]")
    state["i"] = i + 1
    base.scene.dirty = True


reset_play()
print("F = step one frame   G = play/pause   R = reset")


def tick(dt):
    if base.input_manager.is_key_pressed_edge(key.R):
        reset_play()
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
