import numpy as np
from OpenGL.wrapper import none_or_pass

import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp
import one.collider.mj_collider as ocm
import pyglet.window.key as key
from one.grasp.antipodal import antipodal
from one import oum, ouc, ovw, ossop, osso, khi_rs007l, or_2fg7
from one import omppc, ompp

base = ovw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

robot = khi_rs007l.RS007L(pos=(.5, 0, 0))
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# load bunny
bunny = osso.SceneObject.from_file(
    "bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.rgb = (0.8, 0.7, 0.6)
# bunny.alpha = 0.6
bunny.attach_to(base.scene)

# create ground plane
ground = ossop.plane(pos=(0, 0, .01))
ground.attach_to(base.scene)

# setup mj collider
mjc = ocm.MJCollider()
mjc.append(robot)
mjc.append(gripper)
mjc.append(bunny)
mjc.append(ground)
mjc.actors = [robot]
mjc.compile(margin=0.0)

pln_ctx = omppc.PlanningContext(collider=mjc)
planner = ompp.LazyPRMPlanner(pln_ctx=pln_ctx)

# --- compute antipodal grasps on bunny at origin ---
print("Computing antipodal grasps on bunny...")
grasps = antipodal(
    gripper=gripper,
    target_sobj=bunny,
    density=0.01,
    normal_tol_deg=20,
    roll_step_deg=30,
    max_grasps=80
)
print(f"Found {len(grasps)} collision-free grasps")

# --- compute a random stable pose (convex hull) ---
geom = bunny.collisions[0].geom
geom_hull = ogf.convex_hull(geom)
facets = ogs.segment_surface(geom_hull, normal_tol_deg=15)

stable_poses = ogp.compute_stable_poses(
    geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=5.0)

if not stable_poses:
    print("No stable poses found, abort.")
    base.run()

# pick one stable pose at random
pos, rotmat, seg_id, ratio, _ = stable_poses[np.random.randint(len(stable_poses))]
print(f"Selected stable pose: seg={seg_id}, ratio={ratio:.6f}")

# place bunny
bunny.pos = pos
bunny.rotmat = rotmat

# transform grasp poses into world (stable pose)
tf_bunny = oum.tf_from_rotmat_pos(rotmat, pos)

# --- solve IK for each grasp and visualize ---
n_solved = 0
n_failed = 0
goal_qs = None
aux_qs = None

for pose, pre_pose, jaw_width, score in grasps:
    pre_pose_world = tf_bunny @ pre_pose
    pre_tgt_rotmat = pre_pose_world[:3, :3]
    pre_tgt_pos = pre_pose_world[:3, 3]
    qs_list = robot.ik_tcp(tgt_rotmat=pre_tgt_rotmat,
                           tgt_pos=pre_tgt_pos)
    if not qs_list:
        continue

    qs = qs_list[0]
    mjc.set_mecba_qpos(gripper, (jaw_width / 2, jaw_width / 2))
    pln_ctx.set_aux_mecbas(gripper, qs=(jaw_width / 2, jaw_width / 2))
    if not pln_ctx.is_state_valid(qs):
        continue

    goal_qs = qs
    aux_qs= (jaw_width / 2, jaw_width / 2)
    goal_pre_pose = (pre_tgt_pos, pre_tgt_rotmat)
    break

if goal_qs is None:
    # show failed pre-pose
    if grasps:
        pose, pre_pose, jaw_width, score = grasps[0]
        pre_pose_world = tf_bunny @ pre_pose
        ghost = gripper.clone()
        ghost.grip_at(pre_pose_world[:3, 3], pre_pose_world[:3, :3], jaw_width)
        ghost.rgb = (1.0, 0.0, 0.0)
        ghost.alpha = 0.3
        ghost.attach_to(base.scene)
    print("No valid pre-pose IK found.")
    base.run()

# --- plan start -> pre and loop ---
start_qs = robot.qs.copy()
path = None
state = start_qs.copy()
cursor = 0
current_target = goal_qs
tol = 5e-3
move_step = 0.02
need_replan = True
drawn_nodes = {}

# robot.fk(qs=goal_qs)
# gripper.fk(qs=aux_qs)
# base.run()

def move_bunny_once():
    global need_replan, tf_bunny

    moved = False
    pos = np.array(bunny.pos, dtype=np.float32)

    if base.input_manager.is_key_pressed_edge(key.W):
        pos[1] += move_step
        moved = True
    if base.input_manager.is_key_pressed_edge(key.S):
        pos[1] -= move_step
        moved = True
    if base.input_manager.is_key_pressed_edge(key.A):
        pos[0] -= move_step
        moved = True
    if base.input_manager.is_key_pressed_edge(key.D):
        pos[0] += move_step
        moved = True
    rot_step = np.deg2rad(10.0)
    if base.input_manager.is_key_pressed_edge(key.Q):
        rz = oum.rotmat_from_euler(0, 0, rot_step)
        bunny.rotmat = rz @ bunny.rotmat  # world Z
        moved = True
    if base.input_manager.is_key_pressed_edge(key.E):
        rz = oum.rotmat_from_euler(0, 0, -rot_step)
        bunny.rotmat = rz @ bunny.rotmat  # world Z
        moved = True

    if moved:
        bunny.pos = pos
        tf_bunny[:] = oum.tf_from_rotmat_pos(bunny.rotmat, bunny.pos)
        need_replan = True


def clear_drawn():
    for obj in drawn_nodes.values():
        obj.alpha = 0.0


def tick(dt):
    global path, state, current_target, cursor, goal_qs, aux_qs, need_replan

    move_bunny_once()

    if need_replan:
        goal_qs = None
        aux_qs = None
        clear_drawn()
        MAX_DRAW = 5
        count = 0
        for i, (pose, pre_pose, jaw_width, score) in enumerate(grasps):
            pre_pose_world = tf_bunny @ pre_pose
            pre_rot = pre_pose_world[:3, :3]
            pre_pos = pre_pose_world[:3, 3]
            qs_list = robot.ik_tcp(tgt_rotmat=pre_rot, tgt_pos=pre_pos)
            if not qs_list:
                continue
            qs = qs_list[0]
            pln_ctx.set_aux_mecbas(gripper, qs=(jaw_width / 2, jaw_width / 2))
            if not pln_ctx.is_state_valid(qs):
                continue
            if i in drawn_nodes:
                tmp = drawn_nodes[i]
            else:
                tmp = robot.clone()
                tmp.attach_to(base.scene)
                drawn_nodes[i] = tmp

            tmp.rgba = (0.0, 1.0, 0.0, 0.1)
            tmp.fk(qs=qs)
            goal_qs = qs
            aux_qs = (jaw_width / 2, jaw_width / 2)

            count += 1
            if count >= MAX_DRAW:
                break

        if goal_qs is None:
            path = None
            return

        current_target = goal_qs
        path = None
        cursor = 0
        need_replan = False

    if path is None:
        pln_ctx.set_aux_mecbas(gripper, aux_qs)
        path = planner.solve(start=state, goal=current_target)
        if not path:
            return
        gripper.fk(qs=aux_qs)

    next_idx = min(cursor + 1, len(path) - 1)
    state = path[next_idx]
    cursor = next_idx
    robot.fk(qs=state)

    if pln_ctx.states_equal(state, current_target, tol=5e-3):
        if np.allclose(current_target, goal_qs):
            current_target = start_qs
        else:
            current_target = goal_qs
        path = None
        cursor = 0


base.schedule_interval(tick, interval=0.05)
base.run()
