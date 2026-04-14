import time
import numpy as np
import one.physics.mj_env as opme
from one import ouc, oum, ovw, ocm, ossop, omppc, ompr, khi_rs007l

base = ovw.World(cam_pos=(2.2, .7, .7), cam_lookat_pos=(0, 0, .6),
                 toggle_auto_cam_orbit=False)
# world origin
oframe = ossop.frame().attach_to(base.scene)
base_pos1 = np.array([0, 0.5, 0])
base_rotmat = oum.rotmat_from_euler(0, 0, -np.pi / 2)
# robot 1 (left robot)
robot1 = khi_rs007l.RS007L()
robot1.attach_to(base.scene)
robot1.set_rotmat_pos(rotmat=base_rotmat, pos=base_pos1)
robot1.toggle_render_collision = True
# robot 2 (right robot)
robot2 = robot1.clone()
base_pos2 = np.array([0, -0.5, 0])
robot2.attach_to(base.scene)
robot2.set_rotmat_pos(rotmat=base_rotmat, pos=base_pos2)
robot2.toggle_render_collision = True

# robot1
r1s = np.array([1.2637329, 0.72913074, -1.2746582, 1.7030108, -1.2927439, 5.831691], dtype=np.float32)
r1g = np.array([-1.3982087, -1.103838, 1.3426441, -1.7039007, 1.4606364, -2.258566], dtype=np.float32)
# robot2
r2s = np.array([-1.3217797, 2.0543678, 1.3168408, -1.3847662, 1.737289, 0.8488352], dtype=np.float32)
r2g = np.array([-1.2636756, 0.7292364, -1.2745408, 1.4385052, 1.2928442, -5.83168], dtype=np.float32)

collider = ocm.MJCollider()
collider.append(robot1)
collider.append(robot2)
collider.actors = [robot1, robot2]
collider.compile()

pln_ctx = omppc.PlanningContext(collider=collider, cd_step_size=np.pi / 180)
planner = ompr.RRTConnectPlanner(pln_ctx=pln_ctx, extend_step_size=np.pi / 36)
start = np.hstack([r1s, r2s])
goal = np.hstack([r1g, r2g])

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

# print(state_list)
robot1scp = robot1.clone()
robot1scp.fk(qs=r1s)
robot1scp.rgba = (1, 0, 0, 0.5)
robot1scp.attach_to(base.scene)
robot2scp = robot2.clone()
robot2scp.fk(qs=r2s)
robot2scp.rgba = (1, 0, 0, 0.5)
robot2scp.attach_to(base.scene)
robot1gcp = robot1.clone()
robot1gcp.fk(qs=r1g)
robot1gcp.rgba = (0, 0, 1, 0.5)
robot1gcp.attach_to(base.scene)
robot2gcp = robot2.clone()
robot2gcp.fk(qs=r2g)
robot2gcp.rgba = (0, 0, 1, 0.5)
robot2gcp.attach_to(base.scene)

def update_pose(dt, counter, collider):
    if counter[0] < len(state_list):
        sl = collider.get_slice(robot1)
        robot1.fk(qs=state_list[counter[0]][sl])
        sl = collider.get_slice(robot2)
        robot2.fk(qs=state_list[counter[0]][sl])
        counter[0] += 1
    else:
        counter[0] = 0

counter = [0]
base.schedule_interval(update_pose, interval=0.1, counter=counter, collider=collider)
delay = 5.4
base.stop_after(update_pose, delay=delay)

def start_mjenv(dt):
    mjenv = opme.MJEnv(scene=base.scene)
    base.schedule_interval(mjenv.step)

base.schedule_once(start_mjenv, delay=delay + 0.01)

base.run()
