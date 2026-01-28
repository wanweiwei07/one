import builtins
from pyglet.window import key
from one import ovw, ossop, osso, ouc
import one.geom.fitting as ogf
import one.geom.surface as ogs
import one.grasp.placement as ogp

base = ovw.World(cam_pos=(.3, .3, .3), toggle_auto_cam_orbit=True)
builtins.base = base
bunny = osso.SceneObject.from_file(
    "bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.alpha = 0.3
bunny.attach_to(base.scene)
oframe = ossop.gen_frame(length=0.05)
oframe.attach_to(base.scene)

plane_ground = ossop.gen_plane()
plane_ground.attach_to(base.scene)

geom = bunny.collisions[0].geom
geom_hull = ogf.convex_hull(geom)
facets = ogs.segment_surface(geom_hull)
print(facets)
stable_poses = ogp.compute_stable_poses(
    geom_hull.vs, geom_hull.fs, facets, com=None, stable_thresh=5.0)
print(stable_poses)
if not stable_poses:
    print("No stable poses found")

cur_idx = 0
cur_vis = None

def show_pose(idx):
    global cur_vis
    pos, rotmat, seg_id, ratio = stable_poses[idx]
    bunny.pos = pos
    bunny.rotmat = rotmat
    print(f"seg={seg_id}, ratio={ratio:.6f}")

    # optional: visualize current segment
    if cur_vis is not None:
        cur_vis.detach_from(base.scene)
        cur_vis = None
    fs_sub = geom_hull.fs[facets[seg_id]]
    cur_vis = ossop.gen_mesh(geom_hull.vs, fs_sub, rgb=(1, 0, 0), alpha=0.6)
    cur_vis.attach_to(base.scene)
    cur_vis.pos=pos
    cur_vis.rotmat=rotmat

def update(dt):
    global cur_idx
    if not stable_poses:
        return
    if base.input_manager.is_key_pressed_edge(key.SPACE):
        show_pose(cur_idx)
        cur_idx = (cur_idx + 1) % len(stable_poses)

base.schedule_interval(update, interval=0.02)
base.run()
