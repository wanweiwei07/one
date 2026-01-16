import builtins
import numpy as np

from one import oum, ovw, ouc, ossop, ocm, ompsp, ompr, khi_rs007l, or_2fg7

base = ovw.World(cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe = ossop.gen_frame()
oframe.attach_to(base.scene)
robot = khi_rs007l.RS007L()
robot.set_rotmat_pos(pos=(0, 0, 0.01))
gripper = or_2fg7.OR2FG7()
# gripper.open()
robot.engage(gripper)
robot.attach_to(base.scene)
builtins.robot = robot

box = ossop.gen_box(half_extents=(1, .01, .15), pos=(.0, -0.3, .7),
                    collision_type=ouc.CollisionType.AABB)
box.attach_to(base.scene)
box2 = ossop.gen_box(half_extents=(.15, .01, 1), pos=(-.5, -0.3, 0.3),
                     collision_type=ouc.CollisionType.AABB)
box2.attach_to(base.scene)
box3 = ossop.gen_box(half_extents=(.01, 1, .15), pos=(.3, 0.0, 1),
                     collision_type=ouc.CollisionType.AABB)
box3.attach_to(base.scene)
box4 = box2.clone()
box4.pos = (-.5, .3, .5)
box4.attach_to(base.scene)
box5 = box.clone()
box5.pos = (0, .3, 1)
box5.attach_to(base.scene)
plane_ground = ossop.gen_plane()
plane_ground.attach_to(base.scene)

collider = ocm.MJCollider()
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.append(box4)
collider.append(box5)
collider.append(plane_ground)
collider.actors.append(robot)
collider.compile()

jlmt_low = robot.structure.compiled.jlmt_low_by_idx
jlmt_high = robot.structure.compiled.jlmt_high_by_idx
sspp = ompsp.SpaceProvider.from_box_bounds(lmt_low=jlmt_low,
                                           lmt_high=jlmt_high,
                                           collider=collider,
                                           max_edge_step=np.pi / 180)
planner = ompr.RRTConnectPlanner(
    ssp_provider=sspp, step_size=np.pi / 36)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-oum.pi / 2, -oum.pi / 4, oum.pi / 2, -oum.pi / 2, oum.pi / 4, oum.pi / 3])
robot1 = robot.clone()
robot1.fk(qs=start)
robot1.rgba = (1, 0, 0, 0.5)
robot1.attach_to(base.scene)
robot2 = robot.clone()
robot2.fk(qs=goal)
robot2.rgba = (0, 0, 1, 0.5)
robot2.attach_to(base.scene)
counter = [0]
#
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

state_list = planner.solve(
    start=start, goal=goal, verbose=False, max_iters=100000)


def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0


base.schedule_interval(update_pose, interval=0.1, counter=counter)
base.run()
