from one import rm, wd, scn, khi_rs007l

base = wd.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=True)
robot = khi_rs007l.RS007L()
robot.attach_to(base.scene)
robot.toggle_render_collision=True
robot_list = [robot]

def update_pose(dt):
    for robot in robot_list:
        qs = robot.qs + rm.pi / 180
        robot.fk(qs=qs)


def spawn_robot(dt):
    new_robot = robot.clone()
    new_robot.attach_to(base.scene)
    qs = new_robot.qs + rm.pi / 180 * 5 * (len(robot_list))
    new_robot.fk(qs=qs)
    print(new_robot.toggle_render_collision)
    robot_list.append(new_robot)

base.schedule_interval(update_pose, interval=.01)
base.schedule_interval(spawn_robot, interval=3)
base.run()
