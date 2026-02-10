if __name__ == '__main__':
    from one import ovw, ossop, osso

    base = ovw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
    osso.SceneObject.from_file("link6.stl").attach_to(base.scene)
    ossop.frame(length_scale=.3, radius_scale=.3).attach_to(base.scene)
    base.run()