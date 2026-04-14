import builtins
import time
import numpy as np
from one import oum, ovw, ouc, ossop, omppc, ompr, khi_rs007l
from one.collider import SIMDCollider

# Setup viewer
base = ovw.World(
    cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5), toggle_auto_cam_orbit=False
)
builtins.base = base
scene = base.scene

# Add coordinate frame
oframe = ossop.frame()
oframe.attach_to(scene)

# Create robot
robot = khi_rs007l.RS007L()
robot.is_free = True
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(scene)

# Create obstacles (use MESH for SIMD collision)
box = ossop.box(
    half_extents=(1, 0.01, 0.15),
    pos=(0.0, -0.3, 1),
    collision_type=ouc.CollisionType.MESH)
box.attach_to(scene)
box2 = ossop.box(
    half_extents=(.15, .01, 1),
    pos=(-.5, -0.3, 0.5),
    collision_type=ouc.CollisionType.MESH)
box2.attach_to(scene)
box3 = ossop.box(
    half_extents=(.01, 1, .15),
    pos=(.3, 0.0, 1),
    collision_type=ouc.CollisionType.MESH)
box3.attach_to(scene)

# Create SIMD collider
print("Creating SIMDCollider...")
collider = SIMDCollider(use_gpu=True)
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.actors = [robot]
collider.compile()

# Setup motion planner
pln_ctx = omppc.PlanningContext(collider=collider, cd_step_size=np.pi / 180)
planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx, extend_step_size=np.pi / 36)

# Define start and goal
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array(
    [-oum.pi / 2, -oum.pi / 4, oum.pi / 2, -oum.pi / 2, oum.pi / 4, oum.pi / 3]
)

# Run planning
print("\nStarting RRT-Connect planning with SIMDCollider...")
collider.reset_stats()
t0 = time.time()
state_list = planner.solve(
    start=start,
    goal=goal,
    max_iters=3000,
    verbose=True)
t1 = time.time()

# Print results
print(f"\n{'=' * 50}")
print(f"Planning completed in {t1 - t0:.3f}s")
if state_list:
    print(f"Path found with {len(state_list)} waypoints")
else:
    print("No path found")
print(f"{'=' * 50}")

# Print statistics
collider.print_stats()

# Visualization
if state_list:
    # Show start configuration (red, transparent)
    robot1 = robot.clone()
    robot1.fk(qs=start)
    robot1.rgba = (1, 0, 0, 0.5)
    robot1.attach_to(scene)

    # Show goal configuration (blue, transparent)
    robot2 = robot.clone()
    robot2.fk(qs=goal)
    robot2.rgba = (0, 0, 1, 0.5)
    robot2.attach_to(scene)

    # Animate path
    counter = [0]


    def update_pose(dt, counter):
        if counter[0] < len(state_list):
            robot.fk(qs=state_list[counter[0]])
            counter[0] += 1
        else:
            counter[0] = 0


    base.schedule_interval(update_pose, interval=0.1, counter=counter)

base.run()
