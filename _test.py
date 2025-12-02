if __name__ == '__main__':
    import trimesh as trm
    from one import rm, wd, mdl, scn, geom, const, prims

    oframe = prims.gen_frame()
    mesh = trm.load_mesh("bunnysim.stl")
    model = mdl.Model(geom.Mesh(verts=mesh.vertices,
                                faces=mesh.faces,
                                rgb=const.ExtendedColor.LIGHT_STEEL_BLUE))
    print(model.geometry.device_buffer)
    scene = scn.Scene()
    scene.add(model)
    scene.add(oframe)
    base = wd.World(cam_pos=(.3,.3,.3), toggle_auto_cam_orbit=False)
    base.set_scene(scene)
    base.run()
