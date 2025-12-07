if __name__ == '__main__':
    from one import rm, wd, mdl, scn, geom, const, prims, geometry_loader
    import open3d as o3d

    oframe = prims.gen_frame()
    bunny = loader.load_stl("bunny.stl")
    scene = scn.Scene()
    scene.add(bunny)
    scene.add(oframe)
    base = wd.World(cam_pos=(.3,.3,.3), toggle_auto_cam_orbit=True)
    base.set_scene(scene)
    base.run()