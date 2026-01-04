from one import oum, ovw, ossop, khi_rs007l, or_2fg7

base = ovw.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=False)
ossop.gen_frame().attach_to(base.scene)
robot = khi_rs007l.RS007L()
robot.toggle_render_collision = True
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
# grasp first
box = ossop.gen_cylinder(spos=(.3,0,0), epos=(.3,0,.1), radius=.03)
box.attach_to(base.scene)
gripper.grasp(box)
# engage later
robot.engage(gripper)
robot.fk(qs=[0, -oum.pi/4, 0, -oum.pi/2, 0, oum.pi/3])
base.run()