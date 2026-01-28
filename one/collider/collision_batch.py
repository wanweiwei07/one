import numpy as np


class CollisionBatch:
    def __init__(self, items, pairs=None):
        """
        items: list of scene objects (links/sobjs)
        pairs: optional (P,2) int32, default all pairs (i < j)
        """
        self.items = items
        self.vss = None
        self.fss = None
        self.aabb_mins = None
        self.aabb_maxs = None
        self.geom_descs = None
        self.pairs = np.asarray(pairs, dtype=np.uint32)
        self.tfs = np.zeros((len(items), 4, 4), dtype=np.float32)
        self._compile()

    def _compile(self):
        verts_all = []
        faces_all = []
        geom_desc = []
        v_offset = 0
        f_offset = 0
        aabb_min_all = []
        aabb_max_all = []
        for itm in self.items:
            # merge collisions of this obj
            vs_list = []
            fs_list = []
            local_v_offset = 0
            for col in itm.collisions:
                geom = col.geom
                if (geom is None or
                        geom.fs is None):
                    continue
                if (geom.vs is None or
                        geom.vs.size == 0 or
                        geom.fs.size == 0):
                    continue
                tf = col.tf
                vs = (tf[:3, :3] @ geom.vs.T).T + tf[:3, 3]
                fs = geom.fs + local_v_offset
                vs_list.append(vs)
                fs_list.append(fs)
                local_v_offset += vs.shape[0]
            if not vs_list:
                # still record desc to keep indices aligned
                geom_desc.append(
                    [v_offset, 0, f_offset, 0])
                aabb_min_all.append(
                    np.zeros(3, dtype=np.float32))
                aabb_max_all.append(
                    np.zeros(3, dtype=np.float32))
                continue
            itm_vs = np.vstack(vs_list)
            itm_fs = np.vstack(fs_list) + v_offset
            verts_all.append(itm_vs)
            faces_all.append(itm_fs)
            v_count = itm_vs.shape[0]
            f_count = itm_fs.shape[0]
            geom_desc.append([v_offset, v_count,
                              f_offset, f_count])
            v_offset += v_count
            f_offset += f_count
            min_l = itm_vs.min(axis=0)
            max_l = itm_vs.max(axis=0)
            aabb_min_all.append(min_l)
            aabb_max_all.append(max_l)
        self.vss = np.vstack(verts_all).astype(
            np.float32, copy=False) if verts_all \
            else np.zeros((0, 3), np.float32)
        self.fss = np.vstack(faces_all).astype(
            np.int32, copy=False) if faces_all \
            else np.zeros((0, 3), np.int32)
        self.geom_descs = np.array(geom_desc, dtype=np.int32)
        self.aabb_mins = np.vstack(
            aabb_min_all).astype(np.float32, copy=False)
        self.aabb_maxs = np.vstack(
            aabb_max_all).astype(np.float32, copy=False)

    def update_transforms(self):
        for i, itm in enumerate(self.items):
            self.tfs[i] = itm.tf

def compute_wd_obb_batch(local_mins, local_maxs, tfs):
    """
    local_min/local_max: (n,3)
    tf: (N, 4,4) world transform
    return: (wd_centers, half_extents, wd_rotmats)
    """
    centers = (local_mins + local_maxs) * 0.5
    halfs = (local_maxs - local_mins) * 0.5
    rotmats = tfs[:, :3, :3]
    wd_centers = tfs[:, :3, 3]
    wd_cs = np.einsum(
        'nij,nj->ni', rotmats, centers) + wd_centers
    return wd_cs, halfs, rotmats


def compute_wd_aabb_batch(
        local_mins, local_maxs, tfs):
    wd_centers, halfs, rotmats = compute_wd_obb_batch(
        local_mins, local_maxs, tfs)
    wd_halfs = np.einsum(
        'nij,nj->ni', np.abs(rotmats), halfs)
    wd_mins = wd_centers-wd_halfs
    wd_maxs = wd_centers+wd_halfs
    return wd_mins, wd_maxs
