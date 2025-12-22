from one import np, rm, wd, prims, khi_rs007l, or_2fg7

base = wd.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=False)
prims.gen_frame().attach_to(base.scene)
robot = khi_rs007l.RS007L()
robot.attach_to(base.scene)
gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.mount(gripper, np.eye(4))

robot.fk(qs=[0, -rm.pi/4, 0, -rm.pi/2, 0, rm.pi/3])
base.run()