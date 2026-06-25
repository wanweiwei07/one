"""RS007L pick-and-place of a bunny between two FIXED poses, via the library
``gen_pick_place``. The example only sets up the scene (robot, gripper, the
manipulated bunny, the ground and a partition wall) and the pick/place poses;
``one.manipulation.pick_place.gen_pick_place`` does the rest:

    reason a grasp feasible at BOTH poses -> approach+grasp -> lift -> transfer
    -> place -> retreat, returning a MotionData (configs + gripper openings +
    the carried bunny's world pose at every waypoint).

A partition wall stands between pick and place: the transfer routes the HELD
bunny around it, because gen_pick_place collision-checks the carried object (it
rides the gripper link in the carry collider), not just the gripper.

Run:        py -3.12 examples/test_rs007l_pickplace_bunny.py
Headless:   ONE_HEADLESS=1 py -3.12 examples/test_rs007l_pickplace_bunny.py
Keys:       F = step one frame   G = play/pause   R = reset
"""
import os

import numpy as np

import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
from one.grasp.antipodal import antipodal
from one import oum, ouc, ossop, osso, khi_rs007l, or_2fg7  # noqa: F401

HEADLESS = bool(os.environ.get("ONE_HEADLESS"))
BUNNY_STL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bunny.stl")
LIFT = 0.12                      # straight up after grasp / before release (m)
np.random.seed(7)               # antipodal samples randomly; seed for repeatability


# ----------------------------------------------------------------------------
# Scene
robot = khi_rs007l.RS007L()
gripper = or_2fg7.OR2FG7()
robot.mount(gripper, robot.runtime_lnks[-1], update=True)
# the arm's end_effector is DERIVED from this mount (the mech at its flange) --
# no separate assignment; swapping the gripper (re-mount) just follows.
bunny = osso.SceneObject.from_file(BUNNY_STL,
                                   collision_type=ouc.CollisionType.MESH,
                                   is_floating=True)   # manipulated object
ground = ossop.plane(pos=(0, 0, 0.01))
# partition wall between pick and place, across the transfer corridor
wall = ossop.box(pos=(0.0, 0.7, 0.10), xyz_lengths=(0.05, 1.0, 0.3),
                 name="wall", collision_type=ouc.CollisionType.AABB)

# ----------------------------------------------------------------------------
# Pick and place poses: a stable rest pose of the bunny, placed at two spots the
# arm can reach (the pair the shared-grasp demo uses, known to share grasps).
hull = ogf.convex_hull(bunny.collisions[0].geom)
stable = ogp.compute_stable_poses(hull.vs, hull.fs,
                                   ogs.segment_surface(hull), com=None,
                                   stable_thresh=10.0)
assert stable, "no stable pose for bunny"
pos0, rot0 = stable[0][0], stable[0][1]
tf_pick = oum.tf_from_pos_rotmat(pos0 + np.array([-0.5, 0.0, 0.0], np.float32),
                                 rot0)
rot_place = oum.rotmat_from_euler(0, 0, np.deg2rad(45.0)) @ rot0
tf_place = oum.tf_from_pos_rotmat(pos0 + np.array([0.3, 0.3, 0.0], np.float32),
                                  rot_place)

# ----------------------------------------------------------------------------
# Build the collision world ONCE and hold it: robot (+ its mounted gripper) +
# the static ground + wall. The bunny is the manipulated object -- NOT a fixture
# (it is grasped, not an obstacle). It is an explicit value passed to pick_place.
collider = robot.build_collider(fixtures=[ground, wall])

motion = None
for attempt in range(10):
    grasps = antipodal(gripper=gripper, target_sobj=bunny, density=0.006,
                       normal_tol_deg=25, roll_step_deg=30, max_grasps=120)
    if not grasps:
        continue
    # the arm IS the manipulator; the gripper is its end_effector. tcp derived
    # per grasp; the wall is routed around with the bunny carried.
    motion = robot.pick_place(bunny, grasps, tf_pick, tf_place,
                              collider=collider, lift_height=LIFT)
    if motion is not None:
        print(f"attempt {attempt}: {len(grasps)} grasps -> "
              f"{len(motion)} waypoints")
        break
if motion is None:
    raise SystemExit("no feasible pick-and-place plan")

if HEADLESS:
    print("headless OK: pick-and-place planned via gen_pick_place")
    raise SystemExit


# ----------------------------------------------------------------------------
# Viewer + animation. Starts PAUSED; step with F. Each frame just applies the
# precomputed config, gripper opening and carried-object pose -- no mount/unmount.
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
bunny.attach_to(base.scene)

state = {"i": 0, "playing": False}


def show(i):
    robot.fk(qs=motion.robot_qpos_list[i])
    ee_qpos = motion.ee_qpos_list[i]
    if ee_qpos is not None:
        gripper.fk(qs=ee_qpos)             # uniform EE replay (gripper or hand)
    op = motion.obj_pose_list[i]
    if op is not None:
        bunny.pos, bunny.rotmat = op[:3, 3], op[:3, :3]
    base.scene.dirty = True


def step_one():
    if state["i"] >= len(motion):
        state["playing"] = False
        return
    show(state["i"])
    print(f"frame {state['i']}/{len(motion) - 1}")
    state["i"] += 1


def reset_play():
    state["i"], state["playing"] = 0, False
    show(0)


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
