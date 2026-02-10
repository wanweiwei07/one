import builtins, time
import numpy as np
import pyglet.window.key as key
from one import oum, ovw, ouc, ossop, ocm, omppc, ompp, khi_rs007l

base = ovw.World(cam_pos=(-3, 1, 1.5), cam_lookat_pos=(0, 0, 0.5), toggle_auto_cam_orbit=False)
builtins.base = base

oframe = ossop.frame()
oframe.attach_to(base.scene)

robot = khi_rs007l.RS007L()
robot.is_free = True
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

# obstacles
box = ossop.box(half_extents=(.5, .03, .03), pos=(.0, -0.2, 1),
                collision_type=ouc.CollisionType.AABB,
                is_free=True)
box.attach_to(base.scene)

collider = ocm.MJCollider()
collider.append(robot)
collider.append(box)
collider.actors = [robot]
collider.compile()

collider.save("scene_free.xml")

pln_ctx = omppc.PlanningContext(
    collider=collider, cd_step_size=np.pi / 36)
planner = ompp.LazyPRMPlanner(pln_ctx=pln_ctx)

start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-oum.pi / 2, -oum.pi / 4, oum.pi / 2,
                 -oum.pi / 2, oum.pi / 4, oum.pi / 3])

K = 15
EXEC_STEPS = 5

path = None
state = start.copy()
robot.fk(qs=state)
cursor = 0
current_target = goal
tol = 5e-3

# # Run planning with iteration limit
# print("\nStarting RRT-Connect planning with AABBCollider...")
# print("Note: RRT is probabilistic - if planning fails, simply run the script again")
# t0 = time.time()
# state_list = planner.solve(start=state, goal=current_target, verbose=True)
# t1 = time.time()
# # Print results
# print(f"\n{'='*60}")
# print(f"Planning completed in {t1-t0:.3f}s")
# if state_list:
#     print(f"Path found with {len(state_list)} waypoints")
# else:
#     print("No path found")
# print(f"{'='*60}")
#
# for qs in state_list:
#     tmp = robot.clone()
#     tmp.attach_to(base.scene)
#     tmp.fk(qs=qs)
# base.run()

debug_nodes = []


def clear_debug():
    for obj in debug_nodes:
        obj.detach_from(base.scene)
    debug_nodes.clear()


def update_obstacles_by_keys(dt):
    speed = 0.05
    if base.input_manager.is_key_pressed(key.W):
        box.pos = (box.pos[0], box.pos[1], box.pos[2] + speed)
    if base.input_manager.is_key_pressed(key.S):
        box.pos = (box.pos[0], box.pos[1], box.pos[2] - speed)


def is_path_valid(path, cursor, window=K):
    clear_debug()
    return_value = True
    end = min(cursor + window, len(path) - 1)
    for i in range(cursor, end):
        hit = collider.is_collided(path[i])
        tmp = robot.clone()
        tmp.rgba = (1, 0, 0, 0.3) if hit else (0, 1, 0, 0.3)
        tmp.fk(qs=path[i])
        tmp.attach_to(base.scene)
        debug_nodes.append(tmp)
        if hit:
            return_value = False
        if not pln_ctx.is_motion_valid(path[i], path[i + 1]):
            return_value = False
    return return_value


def tick(dt):
    global path, state, current_target, cursor, tol
    update_obstacles_by_keys(dt)
    if pln_ctx.states_equal(
            state, current_target, tol=tol):
        if np.allclose(state, current_target):
            if np.allclose(current_target, goal):
                current_target = start
            else:
                current_target = goal
        path = None
        cursor = 0
    if path is not None:
        if not is_path_valid(path, cursor, K):
            print("not valid")
            path = None
            cursor = 0
    if path is None:
        path = planner.solve(
            start=state, goal=current_target)
        if not path:
            return
    # next_idx = min(cursor + EXEC_STEPS, len(path) - 1)
    next_idx = min(cursor + 1, len(path) - 1)
    state = path[next_idx]
    cursor = next_idx
    robot.fk(qs=state)


base.schedule_interval(tick, interval=0.2)
base.run()
