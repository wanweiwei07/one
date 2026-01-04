from one import oum, ovw, ouc, ossop, osso

oframe = ossop.gen_frame()
bunny = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.CAPSULE)
bunny.toggle_render_collision = True
base = ovw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
oframe.attach_to(base.scene)
bunny.attach_to(base.scene)

bunny2 = bunny.clone()
bunny2.pos = bunny.pos + oum.np.array([0, 0.1, 0])
bunny2.attach_to(base.scene)
base.run()
