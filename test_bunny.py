if __name__ == '__main__':
    from one import rm, wd, mdl, scn, geom, const, prims, sob

    oframe = prims.gen_frame()
    bunny = sob.SceneObject.from_file("bunny.stl")
    base = wd.World(cam_pos=(.3,.3,.3), toggle_auto_cam_orbit=True)
    oframe.attach_to(base.scene)
    bunny.attach_to(base.scene)
    base.run()