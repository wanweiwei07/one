from one import rm, wd, mdl, scn, geom, const, prims, sob

oframe = prims.gen_frame()
bunny = sob.SceneObject.from_file("bunny.stl", collision_type=const.CollisionType.CAPSULE)
bunny.toggle_render_collision = True
base = wd.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
oframe.attach_to(base.scene)
bunny.attach_to(base.scene)

bunny2 = bunny.clone()
bunny2.pos = bunny.pos + rm.np.array([0, 0.1, 0])
bunny2.attach_to(base.scene)
base.run()
