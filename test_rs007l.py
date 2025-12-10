
from one import wd, scn, prims, khi_rs007l

base = wd.World(cam_pos=(1, .3, 1), cam_lookat_pos=(0,0,.5),
                toggle_auto_cam_orbit=True)
scene = scn.Scene()
robot = khi_rs007l.RS007L()
robot.attach_to(scene)
base.set_scene(scene)
base.run()