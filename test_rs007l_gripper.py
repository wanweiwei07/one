from one import rm, wd, scn, prims, khi_rs007l

base = wd.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=False)
scene = scn.Scene()
prims.gen_frame().attach_to(scene)
robot = khi_rs007l.RS007L()
robot.attach_to(scene)
robot_list = [robot]
base.set_scene(scene)
base.run()