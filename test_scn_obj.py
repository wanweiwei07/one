if __name__ == '__main__':
    from one import rm, wd, mdl, scn, geom, const, prims, sob

    oframe = prims.gen_frame()
    bunny = sob.SceneObject.from_file("bunny.stl")
    scene = scn.Scene()
    scene.add(bunny)
    scene.add(oframe)
    base = wd.World(cam_pos=(.3,.3,.3), toggle_auto_cam_orbit=False)
    base.set_scene(scene)
    base.run()