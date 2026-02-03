import builtins
import time
import numpy as np
from one import oum, ovw, ouc, ossop, omppc, ompr, khi_rs007l
from one.collider import GPUAABBCollider

# Setup viewer
base = ovw.World(cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5), 
                 toggle_auto_cam_orbit=False)
builtins.base = base

# Add coordinate frame
oframe = ossop.gen_frame()
oframe.attach_to(base.scene)

# Create robot
robot = khi_rs007l.RS007L()
robot.is_free = True
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

# Create obstacles (use MESH for consistency with SIMD test)
# Make box2 thicker (Y from .01 to .05) to increase collision likelihood
box = ossop.gen_box(half_extents=(1, .05, .15), pos=(.0, -0.3, 1.0),
                    collision_type=ouc.CollisionType.MESH)
box.rgba = (0.8, 0.8, 0.8, 0.3)  # Gray, semi-transparent
box.attach_to(base.scene)
box2 = ossop.gen_box(half_extents=(.15, .05, 1), pos=(-.5, -0.3, 0.5),
                     collision_type=ouc.CollisionType.MESH)
box2.rgba = (1, 0, 0, 0.3)  # Red, semi-transparent
box2.attach_to(base.scene)
box3 = ossop.gen_box(half_extents=(.05, 1, .15), pos=(.3, 0.0, 1.0),
                     collision_type=ouc.CollisionType.MESH)
box3.rgba = (0.8, 0.8, 0.8, 0.3)  # Gray, semi-transparent
box3.attach_to(base.scene)
base.run()
# Create AABB collider
print("Creating AABBCollider...")
collider = GPUAABBCollider()
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.actors = [robot]
collider.compile()

pln_ctx = omppc.PlanningContext(collider=collider)
planner = ompr.RRTConnectPlanner(
    pln_ctx=pln_ctx, extend_step_size=np.pi / 36)

# Define start and goal (same as original test)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-oum.pi / 2, -oum.pi / 4, oum.pi / 2, 
                 -oum.pi / 2, oum.pi / 4, oum.pi / 3])

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

def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0

base.schedule_interval(update_pose, interval=0.1, counter=counter)
base.run()
