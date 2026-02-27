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

base = ovw.World(
    cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, 0.75), toggle_auto_cam_orbit=False
)
ossop.frame().attach_to(base.scene)

robot = khi_rs007l.RS007L()
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# load bunny1
bunny = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.rgb = (0.8, 0.7, 0.6)
bunny.attach_to(base.scene)

# load bunny2
bunny2 = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny2.rgb = (0.7, 0.8, 0.6)
bunny2.attach_to(base.scene)

# create ground plane
ground = ossop.plane(pos=(0, 0, 0.01))
ground.attach_to(base.scene)

# setup mj collider
mjc = ocm.MJCollider()
mjc.append(robot)
mjc.append(gripper)
mjc.append(bunny)
mjc.append(bunny2)
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
    max_grasps=80,
)
print(f"Found {len(grasps)} collision-free grasps")

# --- compute stable poses for both bunnies ---
geom = bunny.collisions[0].geom
geom_hull = ogf.convex_hull(geom)
facets = ogs.segment_surface(geom_hull)

stable_poses = ogp.compute_stable_poses(
    geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=10.0
)

print(f"Found {len(stable_poses)} stable poses")
if not stable_poses:
    print("No stable poses found, abort.")
    base.run()

# place bunny1
pos, rotmat, seg_id, ratio, _ = stable_poses[0]
print(f"Selected stable pose for bunny1: seg={seg_id}, ratio={ratio:.6f}")
pos += np.array([-0.5, 0.0, 0.0], dtype=np.float32)
bunny.pos = pos
bunny.rotmat = rotmat
tf_bunny = oum.tf_from_rotmat_pos(rotmat, pos)

# place bunny2 with random offset
pos2, rotmat2, seg_id2, ratio2, _ = stable_poses[0]
print(f"Selected stable pose for bunny2: seg={seg_id2}, ratio={ratio2:.6f}")
pos2 += np.array([0.3, 0.3, 0.0], dtype=np.float32)
rotmat2 = oum.rotmat_from_euler(0, 0, np.deg2rad(45.0)) @ rotmat2
bunny2.pos = pos2
bunny2.rotmat = rotmat2
tf_bunny2 = oum.tf_from_rotmat_pos(rotmat2, pos2)

# dump all pre-grasp candidates for standalone IK diagnosis
pre_pose_pos_list = []
pre_pose_rot_list = []
jaw_width_list = []
for pose, pre_pose, jaw_width, score in grasps:
    gl_pre_pose = tf_bunny @ pre_pose
    pre_pose_pos_list.append(gl_pre_pose[:3, 3].astype(np.float32))
    pre_pose_rot_list.append(gl_pre_pose[:3, :3].astype(np.float32))
    jaw_width_list.append(np.float32(jaw_width))
if len(pre_pose_pos_list) > 0:
    np.savez(
        "rs007l_grasp_candidates.npz",
        pre_pos=np.asarray(pre_pose_pos_list, dtype=np.float32),
        pre_rot=np.asarray(pre_pose_rot_list, dtype=np.float32),
        jaw_width=np.asarray(jaw_width_list, dtype=np.float32),
    )
    print(f"Saved {len(pre_pose_pos_list)} candidates to rs007l_grasp_candidates.npz")

# --- solve IK for each grasp and visualize ---
n_solved = 0
n_failed = 0
goal_qs = None
aux_qs = None

for pose, pre_pose, jaw_width, score in grasps:
    gl_pre_pose = tf_bunny @ pre_pose
    pre_tgt_rotmat = gl_pre_pose[:3, :3]
    pre_tgt_pos = gl_pre_pose[:3, 3]
    qs = robot.ik_tcp_nearest(tgt_rotmat=pre_tgt_rotmat, tgt_pos=pre_tgt_pos)
    if qs is None:
        continue
    mjc.set_mecba_qpos(gripper, (jaw_width / 2, jaw_width / 2))
    pln_ctx.set_aux_mecbas(gripper, qs=(jaw_width / 2, jaw_width / 2))
    if not pln_ctx.is_state_valid(qs):
        continue

    goal_qs = qs
    aux_qs = (jaw_width / 2, jaw_width / 2)
    goal_pre_pose = (pre_tgt_pos, pre_tgt_rotmat)
    break


if goal_qs is None:
    # show failed pre-pose
    if grasps:
        pose, pre_pose, jaw_width, score = grasps[0]
        gl_pre_pose = tf_bunny @ pre_pose
        ghost = gripper.clone()
        ghost.grip_at(gl_pre_pose[:3, 3], gl_pre_pose[:3, :3], jaw_width)
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
drawn_nodes_bunny1 = {}
drawn_nodes_bunny2 = {}
drawn_nodes_shared = {}

# robot.fk(qs=goal_qs)
# gripper.fk(qs=aux_qs)
# base.run()


def move_bunny_once():
    global need_replan, tf_bunny, tf_bunny2

    moved = False
    pos = np.array(bunny.pos, dtype=np.float32)
    pos2 = np.array(bunny2.pos, dtype=np.float32)

    if base.input_manager.is_key_pressed(key.W):
        pos[1] += move_step
        moved = True
    if base.input_manager.is_key_pressed(key.S):
        pos[1] -= move_step
        moved = True
    if base.input_manager.is_key_pressed(key.A):
        pos[0] -= move_step
        moved = True
    if base.input_manager.is_key_pressed(key.D):
        pos[0] += move_step
        moved = True
    rot_step = np.deg2rad(10.0)
    if base.input_manager.is_key_pressed(key.Q):
        rz = oum.rotmat_from_euler(0, 0, rot_step)
        bunny.rotmat = rz @ bunny.rotmat
        moved = True
    if base.input_manager.is_key_pressed(key.E):
        rz = oum.rotmat_from_euler(0, 0, -rot_step)
        bunny.rotmat = rz @ bunny.rotmat
        moved = True

    if base.input_manager.is_key_pressed(key.I):
        pos2[1] += move_step
        moved = True
    if base.input_manager.is_key_pressed(key.K):
        pos2[1] -= move_step
        moved = True
    if base.input_manager.is_key_pressed(key.J):
        pos2[0] -= move_step
        moved = True
    if base.input_manager.is_key_pressed(key.L):
        pos2[0] += move_step
        moved = True
    if base.input_manager.is_key_pressed(key.U):
        rz = oum.rotmat_from_euler(0, 0, rot_step)
        bunny2.rotmat = rz @ bunny2.rotmat
        moved = True
    if base.input_manager.is_key_pressed(key.O):
        rz = oum.rotmat_from_euler(0, 0, -rot_step)
        bunny2.rotmat = rz @ bunny2.rotmat
        moved = True

    if moved:
        bunny.pos = pos
        bunny2.pos = pos2
        tf_bunny[:] = oum.tf_from_rotmat_pos(bunny.rotmat, bunny.pos)
        tf_bunny2[:] = oum.tf_from_rotmat_pos(bunny2.rotmat, bunny2.pos)
        need_replan = True


def clear_drawn():
    for obj in drawn_nodes_bunny1.values():
        obj.alpha = 0.0
    for obj in drawn_nodes_bunny2.values():
        obj.alpha = 0.0
    for gripper1, gripper2 in drawn_nodes_shared.values():
        gripper1.alpha = 0.0
        gripper2.alpha = 0.0


def tick(dt):
    global path, state, current_target, cursor, goal_qs, aux_qs, need_replan

    move_bunny_once()

    if need_replan:
        goal_qs = None
        aux_qs = None
        clear_drawn()
        MAX_DRAW = 30
        
        feasible_bunny1 = {}
        feasible_bunny2 = {}
        
        for i, (pose, pre_pose, jaw_width, score) in enumerate(grasps):
            pre_pose_world1 = tf_bunny @ pre_pose
            pre_rot1 = pre_pose_world1[:3, :3]
            pre_pos1 = pre_pose_world1[:3, 3]
            qs_list1 = robot.ik_tcp(tgt_rotmat=pre_rot1, tgt_pos=pre_pos1)
            if qs_list1:
                qs1 = qs_list1[0]
                pln_ctx.set_aux_mecbas(gripper, qs=(jaw_width / 2, jaw_width / 2))
                if pln_ctx.is_state_valid(qs1):
                    feasible_bunny1[i] = (qs1, jaw_width)
            
            pre_pose_world2 = tf_bunny2 @ pre_pose
            pre_rot2 = pre_pose_world2[:3, :3]
            pre_pos2 = pre_pose_world2[:3, 3]
            qs_list2 = robot.ik_tcp(tgt_rotmat=pre_rot2, tgt_pos=pre_pos2)
            if qs_list2:
                qs2 = qs_list2[0]
                pln_ctx.set_aux_mecbas(gripper, qs=(jaw_width / 2, jaw_width / 2))
                if pln_ctx.is_state_valid(qs2):
                    feasible_bunny2[i] = (qs2, jaw_width)
        
        shared_indices = set(feasible_bunny1.keys()) & set(feasible_bunny2.keys())
        
        count1 = 0
        for i in sorted(feasible_bunny1.keys()):
            qs, jaw_width = feasible_bunny1[i]
            if i in drawn_nodes_bunny1:
                tmp = drawn_nodes_bunny1[i]
            else:
                tmp = robot.clone()
                tmp.attach_to(base.scene)
                drawn_nodes_bunny1[i] = tmp
            tmp.rgba = (0.0, 1.0, 0.0, 0.1)
            tmp.fk(qs=qs)
            count1 += 1
            if count1 >= MAX_DRAW:
                break
        
        count2 = 0
        for i in sorted(feasible_bunny2.keys()):
            qs, jaw_width = feasible_bunny2[i]
            if i in drawn_nodes_bunny2:
                tmp = drawn_nodes_bunny2[i]
            else:
                tmp = robot.clone()
                tmp.attach_to(base.scene)
                drawn_nodes_bunny2[i] = tmp
            tmp.rgba = (0.0, 1.0, 0.0, 0.1)
            tmp.fk(qs=qs)
            count2 += 1
            if count2 >= MAX_DRAW:
                break
        
        count_shared = 0
        for i in sorted(shared_indices):
            pose, pre_pose, jaw_width, score = grasps[i]
            
            pre_pose_world1 = tf_bunny @ pre_pose
            pre_rot1 = pre_pose_world1[:3, :3]
            pre_pos1 = pre_pose_world1[:3, 3]
            
            pre_pose_world2 = tf_bunny2 @ pre_pose
            pre_rot2 = pre_pose_world2[:3, :3]
            pre_pos2 = pre_pose_world2[:3, 3]

            # Ensure: every shared (blue) grasp has corresponding feasible (green) robot poses.
            qs1, _ = feasible_bunny1[i]
            qs2, _ = feasible_bunny2[i]
            if i in drawn_nodes_bunny1:
                tmp1 = drawn_nodes_bunny1[i]
            else:
                tmp1 = robot.clone()
                tmp1.attach_to(base.scene)
                drawn_nodes_bunny1[i] = tmp1
            tmp1.rgba = (0.0, 1.0, 0.0, 0.1)
            tmp1.fk(qs=qs1)

            if i in drawn_nodes_bunny2:
                tmp2 = drawn_nodes_bunny2[i]
            else:
                tmp2 = robot.clone()
                tmp2.attach_to(base.scene)
                drawn_nodes_bunny2[i] = tmp2
            tmp2.rgba = (0.0, 1.0, 0.0, 0.1)
            tmp2.fk(qs=qs2)
            
            if i in drawn_nodes_shared:
                gripper1, gripper2 = drawn_nodes_shared[i]
            else:
                gripper1 = gripper.clone()
                gripper1.attach_to(base.scene)
                gripper2 = gripper.clone()
                gripper2.attach_to(base.scene)
                drawn_nodes_shared[i] = (gripper1, gripper2)
            
            gripper1.rgba = (0.0, 0.0, 1.0, 0.3)
            gripper1.grip_at(pre_pos1, pre_rot1, jaw_width)
            
            gripper2.rgba = (0.0, 0.0, 1.0, 0.3)
            gripper2.grip_at(pre_pos2, pre_rot2, jaw_width)
            
            if goal_qs is None:
                goal_qs = qs1
                aux_qs = (jaw_width / 2, jaw_width / 2)
            count_shared += 1
            if count_shared >= MAX_DRAW:
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
