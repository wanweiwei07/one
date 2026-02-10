import builtins, time
import numpy as np
from one import oum, ovw, ouc, ossop, ocm, omppc, ompr, khi_rs007l

base = ovw.World(cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe = ossop.frame()
oframe.attach_to(base.scene)
robot = khi_rs007l.RS007L()
robot.is_free=True
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

box = ossop.box(half_extents=(1, .01, .15), pos=(.0, -0.3, 1),
                collision_type=ouc.CollisionType.AABB)
box.attach_to(base.scene)
box2 = ossop.box(half_extents=(.15, .01, 1), pos=(-.5, -0.3, 0.5),
                 collision_type=ouc.CollisionType.AABB)
box2.attach_to(base.scene)
box3 = ossop.box(half_extents=(.01, 1, .15), pos=(.3, 0.0, 1),
                 collision_type=ouc.CollisionType.AABB)
box3.attach_to(base.scene)

collider = ocm.MJCollider()
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.actors = [robot]
collider.compile()

pln_ctx = omppc.PlanningContext(collider=collider, cd_step_size=np.pi / 180)
planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx, extend_step_size=np.pi / 36)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-oum.pi / 2, -oum.pi / 4, oum.pi / 2, -oum.pi / 2, oum.pi / 4, oum.pi / 3])

# Run planning with iteration limit
print("\nStarting RRT-Connect planning with AABBCollider...")
print("Note: RRT is probabilistic - if planning fails, simply run the script again")
t0 = time.time()
state_list = planner.solve(start=start, goal=goal, verbose=False, max_iters=3000)
t1 = time.time()
# Print results
print(f"\n{'='*60}")
print(f"Planning completed in {t1-t0:.3f}s")
if state_list:
    print(f"Path found with {len(state_list)} waypoints")
else:
    print("No path found")
print(f"{'='*60}")

robot1 = robot.clone()
robot1.fk(qs=start)
robot1.rgba = (1, 0, 0, 0.5)
robot1.attach_to(base.scene)
robot2 = robot.clone()
robot2.fk(qs=goal)
robot2.rgba = (0, 0, 1, 0.5)
robot2.attach_to(base.scene)

counter = [0]

# rrt_iter = planner.solve_iter(start=start, goal=goal,
#                               verbose=True, max_iters=100000)
# rrt_states = []
# final_path = None
#
# def update_pose(dt):
#     global final_path
#     if final_path is not None:
#         return
#     try:
#         status, data, tree = next(rrt_iter)
#         if status == "extend_start" or status == "extend_goal":
#             qs = data
#             robot.fk(qs=qs)
#         elif status == "success":
#             final_path = data
#             print("RRT finished, path length =", len(final_path))
#         elif status == "failed":
#             print("RRT failed")
#     except StopIteration:
#         pass
#
# base.schedule_interval(update_pose, interval=0.05)
# base.run()

def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0


base.schedule_interval(update_pose, interval=0.1, counter=counter)
base.run()
