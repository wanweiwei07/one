if __name__ == '__main__':
    import builtins
    from one import rm, wd, mdl, scn, geom, const, prims, sob

    base = wd.World(cam_pos=(.3,.3,.3), toggle_auto_cam_orbit=False)
    scene = scn.Scene()
    builtins.scene = scene
    bunny = sob.SceneObject.from_file("link6.stl")
    # oframe = prims.gen_frame()
    # scene.add(oframe)
    scene.add(bunny)
    base.set_scene(scene)
    base.run()
