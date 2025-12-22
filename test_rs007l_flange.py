if __name__ == '__main__':
    from one import wd, prims, sob

    base = wd.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
    sob.SceneObject.from_file("link6.stl").attach_to(base.scene)
    prims.gen_frame(length_scale=.3, radius_scale=.3).attach_to(base.scene)
    base.run()
