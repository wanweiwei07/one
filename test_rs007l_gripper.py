from one import rm, wd, scn, prims, khi_rs007l, or_2fg7

base = wd.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=False)
scene = scn.Scene()
prims.gen_frame().attach_to(scene)
robot = khi_rs007l.RS007L()
robot.attach_to(scene)
# gripper = or_2fg7.OR2FG7()
# gripper.attach_to(scene)
# robot.mount(gripper, rm.eye(4))
base.set_scene(scene)
base.run()