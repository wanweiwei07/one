import numpy as np
import one.scene.geometry_operation as osgop
import one.scene.collision_shape as osc


def _ray_to_local(orig, direction, tf):
    inv_tf = np.linalg.inv(tf)
    orig_l = (inv_tf @ np.append(orig, 1.0))[:3]
    dir_l = inv_tf[:3, :3] @ direction
    return orig_l, dir_l


def ray_shoot_scene_object(scene_obj, orig, direction):
    """
    Ray cast against all collision shapes of a SceneObject.
    orig, direction are in world coordinates.
    Returns:
        hit_pos, hit_n, hit_t, hit_cid, hit_fid
        (all numpy arrays, sorted by hit_t). Returns None if no hit.
    """
    node_tf = scene_obj.node.wd_tf
    all_pos = []
    all_n = []
    all_t = []
    all_c = []
    all_f = []
    for cidx, c in enumerate(scene_obj.collisions):
        col_tf = node_tf @ c._tf
        orig_l, dir_l = _ray_to_local(orig, direction, col_tf)
        if isinstance(c, osc.MeshCollisionShape):
            res = osgop.ray_shoot_flat(
                orig_l, dir_l,
                c.geometry._vs,
                c.geometry._fs,
                c.geometry._fns)
            if res is None:
                continue
            hit_pos, hit_n, hit_t, hit_fid = res
            if hit_t.size == 0:
                continue
            rot = col_tf[:3, :3]
            pos = (rot @ hit_pos.T).T + col_tf[:3, 3]
            nrm = (rot @ hit_n.T).T
            c_ids = np.full(hit_t.shape[0], cidx, dtype=np.int32)
            all_pos.append(pos)
            all_n.append(nrm)
            all_t.append(hit_t)
            all_c.append(c_ids)
            all_f.append(hit_fid)
    if not all_t:
        return None
    hit_pos = np.vstack(all_pos)
    hit_n = np.vstack(all_n)
    hit_t = np.concatenate(all_t)
    hit_cid = np.concatenate(all_c)
    hit_fid = np.concatenate(all_f)
    order = np.argsort(hit_t)
    return (hit_pos[order],
            hit_n[order],
            hit_t[order],
            hit_cid[order],
            hit_fid[order])
