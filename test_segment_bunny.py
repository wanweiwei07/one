import one.geom.fitting as ogf
import one.geom.surface as ogs
from one import ovw, ossop, osso, ouc, key

base = ovw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)

oframe = ossop.gen_frame()
bunny = osso.SceneObject.from_file(
    "bunny.stl", collision_type=ouc.CollisionType.MESH)
oframe.attach_to(base.scene)
bunny.attach_to(base.scene)
bunny.alpha=.3

geom_hull = ogf.convex_hull(bunny.collisions[0].geom)
segmented = ogs.segment_surface(geom_hull, normal_tol_deg=5)
for fids in segmented:
    ossop.gen_mesh(geom_hull.vs, geom_hull.fs[fids],
                   rgb=(1, 0, 0)).attach_to(base.scene)
    break

counter = [0]
def draw_segmented(dt, geom, segmented, counter):
    if counter[0] >= len(segmented):
        return
    if base.input_manager.is_key_pressed(key.SPACE):
        fids = segmented[counter[0]]
        ossop.gen_mesh(geom.vs, geom.fs[fids],
                       rgb=(1, 0, 0)).attach_to(base.scene)
        counter[0] += 1

base.schedule_interval(draw_segmented, 0.01, geom_hull, segmented, counter)
base.run()