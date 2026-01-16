import builtins
import numpy as np
from one import oum, ovw, ouc, ossop, ocm, ompsp, ompp, khi_rs007l

base = ovw.World(cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5))
builtins.base = base
oframe = ossop.gen_frame()
oframe.attach_to(base.scene)
robot = khi_rs007l.RS007L()
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

box = ossop.gen_box(half_extents=(1, .01, .15), pos=(.0, -0.3, 1),
                    collision_type=ouc.CollisionType.AABB)
box.attach_to(base.scene)
box2 = ossop.gen_box(half_extents=(.15, .01, 1), pos=(-.5, -0.3, 0.5),
                    collision_type=ouc.CollisionType.AABB)
box2.attach_to(base.scene)
box3 = ossop.gen_box(half_extents=(.01, 1, .15), pos=(.3, 0.0, 1),
                    collision_type=ouc.CollisionType.AABB)
box3.attach_to(base.scene)

collider = ocm.MJCollider()
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.actors = [robot]
collider.compile()

jlmt_low = robot.structure.compiled.jlmt_low_by_idx
jlmt_high = robot.structure.compiled.jlmt_high_by_idx
sspp = ompsp.SpaceProvider.from_box_bounds(lmt_low=jlmt_low,
                                           lmt_high=jlmt_high,
                                           collider=collider,
                                           max_edge_step=np.pi/36)
# planner = ompp.PRMPlanner(ssp_provider=sspp)
planner = ompp.LazyPRMPlanner(ssp_provider=sspp)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-oum.pi / 2, -oum.pi / 4, oum.pi / 2, -oum.pi / 2, oum.pi / 4, oum.pi / 3])
state_list = planner.solve(start=start, goal=goal)
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
