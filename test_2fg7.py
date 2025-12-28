import numpy as np
from one import wd, const, prims, or_2fg7

base = wd.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                toggle_auto_cam_orbit=True)
oframe = prims.gen_frame().attach_to(base.scene)
gripper = or_2fg7.OR2FG7()

gripper.toggle_render_collision = True
gripper.attach_to(base.scene)
gripper.fk()

box = prims.gen_cylinder(spos=(.3, 0, 0), epos=(.3, 0, .1), radius=.03,
                         collision_type=const.CollisionType.AABB)
box.attach_to(base.scene)
box.toggle_render_collision = True

gripper.grasp(box)

# robot_list = [robot]
#
# def update_pose(dt):
#     print(len(robot_list))
#     for robot in robot_list:
#         qs = robot.qs+0.001
#         robot.fk(qs=qs)
#         robot.update()
# #
# def spawn_robot(dt):
#     new_robot = robot.clone()
#     new_robot.attach_to(base.scene)
#     qs = new_robot.qs+.001*5*(len(robot_list))
#     new_robot.fk(qs=qs)
#     robot_list.append(new_robot)
# #
# base.schedule_interval(update_pose, interval=.01)
# base.schedule_interval(spawn_robot, interval=3)
base.run()
