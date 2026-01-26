import one.scene.geometry_operation as osgop
from one import ovw, ossop, osso, ouc

base = ovw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)

oframe = ossop.gen_frame()
bunny = osso.SceneObject.from_file(
    "bunny.stl", collision_type=ouc.CollisionType.MESH)
oframe.attach_to(base.scene)
bunny.attach_to(base.scene)
bunny.alpha=.3

geom = bunny.collisions[0].geometry
fs_subs = osgop.segment_surface(geom, normal_tol_deg=180)
for fs_sub in fs_subs:
    ossop.create_from_vfs(geom.vs, geom.fs[fs_sub],
                          rgb=(1, 0, 0)).attach_to(base.scene)

base.run()