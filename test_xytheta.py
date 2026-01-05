from one import ovw, xyt, ossop, ouc

base = ovw.World(cam_pos=(3, 1, 2), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
oframe = ossop.gen_frame()
oframe.attach_to(base.scene)
xyt_bot = xyt.XYThetaRobot()
xyt_bot.rgb=ouc.ExtendedColor.LAWN_GREEN
xyt_bot.attach_to(base.scene)
base.run()