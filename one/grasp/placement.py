import numpy as np
import one.utils.math as oum
import one.utils.helper as ouh
import one.geom.fitting as ogf
import one.geom.ops2d as ogo2d
import one.scene.scene_object_primitive as ossop


def compute_stable_poses(
        vs, fs, facets, com=None, stable_thresh=5.0):
    """segs = list of list of face ids"""
    if com is None:
        com = vs.mean(axis=0)
    com = np.asarray(com, dtype=np.float32)
    stable_poses = []
    for seg_id, fids in enumerate(facets):
        fids = np.asarray(fids, dtype=np.int32)
        if fids.size == 0:
            continue
        # fit plane
        fs_sub = fs[fids]
        vids = np.unique(fs_sub.reshape(-1))
        n, d = ogf.fit_plane_from_pts(vs[vids])
        # projection point
        dist = np.dot(n, com) + d
        proj_pt3d = com - n * dist
        # boundary polygon
        polygon_vids = ogo2d.extract_boundary(fs_sub)
        if len(polygon_vids) == 0:
            continue
        # build 2D coords on plane
        frame = oum.frame_from_normal(n)
        poly_pts3d = vs[polygon_vids]
        poly_pts2d = np.stack(
            [poly_pts3d @ frame[:, 0],
             poly_pts3d @ frame[:, 1]], axis=1)
        proj_pt2d = np.array(
            [proj_pt3d @ frame[:, 0],
             proj_pt3d @ frame[:, 1]],
            dtype=np.float32)
        # inside check
        if not ogo2d.pts_in_polygon2d(proj_pt2d, poly_pts2d):
            continue
        # boundary segments in 2D
        line_segs = np.stack([poly_pts2d, np.roll(
            poly_pts2d, -1, axis=0)], axis=1)
        min_dist, min_pt2d = ogo2d.mindist_to_linesegs2d(
            proj_pt2d, line_segs)
        if not np.isfinite(min_dist) or min_dist <= 1e-12:
            continue
        # ratio
        h = np.linalg.norm(com - proj_pt3d)
        ratio = h / min_dist
        print(ratio)
        if ratio >= stable_thresh:
            continue
        # ossop.gen_mesh(vs, fs_sub, rgb=ouh.rand_rgb()).attach_to(base.scene)
        # ossop.gen_sphere(proj_pt3d, 0.002, rgb=[1, 0, 0]).attach_to(base.scene)
        # ossop.gen_sphere(com, 0.002, rgba=[0, 0, 1]).attach_to(base.scene)
        # base.run()
        # recover q3 in plane
        min_pt3d = (proj_pt3d + frame[:, 0] * min_pt2d[0] +
                    frame[:, 1] * min_pt2d[1])
        # build pose
        z = com - proj_pt3d
        z = z / (np.linalg.norm(z) + oum.eps)
        y = min_pt3d - proj_pt3d
        y = y / (np.linalg.norm(y) + oum.eps)
        x = np.cross(y, z)
        x = x / (np.linalg.norm(x) + oum.eps)
        rotmat = np.column_stack((x, y, z)).astype(np.float32)
        stable_poses.append((-(proj_pt3d @ rotmat).astype(np.float32),
                             rotmat.T, seg_id, ratio))
    #  sort by ratio (smaller = more stable)
    stable_poses.sort(key=lambda x: x[3])
    return stable_poses
