import numpy as np
import one.utils.math as oum
import one.viewer.device_buffer as ovdb


class Geometry:

    def __init__(self,
                 verts,
                 faces,
                 per_vert_rgbs=None):
        if faces is not None:
            self._vs, self._fs = self._merge_vs_and_fs(verts, faces)
            self._fns, self._vns = self._compute_vns()
        else:
            self._vs = np.asarray(verts, dtype=np.float32)
            self._fs = None
            self._fns = None
            self._vns = None
            self.per_vert_rgbs = per_vert_rgbs
        self._device_buffer = None

    def get_device_buffer(self):
        if self._device_buffer is None:
            if self._fs is None:
                self._device_buffer = ovdb.PointCloudBuffer(
                    self._vs, self.per_vert_rgbs)
            else:
                self._device_buffer = ovdb.MeshBuffer(
                    self._vs, self._fs, self._vns)
        return self._device_buffer

    @property
    def vs(self):  # verts
        return self._vs

    @property
    def fs(self):  # faces
        return self._fs

    @property
    def vns(self):  # vertex normals
        return self._vns

    @property
    def fns(self):  # face normals
        return self._fns

    def _compute_vns(self):
        v1 = self._vs[self._fs[:, 1]] - self._vs[self._fs[:, 0]]
        v2 = self._vs[self._fs[:, 2]] - self._vs[self._fs[:, 0]]
        raw_fns = np.cross(v1, v2).astype(np.float32)
        _, unit_fns = oum.unit_vec(raw_fns)
        # vert normals
        raw_vns = np.zeros_like(self._vs)
        np.add.at(raw_vns, self._fs[:, 0], unit_fns)
        np.add.at(raw_vns, self._fs[:, 1], unit_fns)
        np.add.at(raw_vns, self._fs[:, 2], unit_fns)
        _, unit_vns = oum.unit_vec(raw_vns)
        return unit_fns, unit_vns

    def _merge_vs_and_fs(self, vs, fs, tol=1e-6):
        q = np.round(vs / tol).astype(np.int64)
        unique_q, inv = np.unique(q, axis=0, return_inverse=True)
        new_vs = np.zeros((len(unique_q), 3), dtype=vs.dtype)
        np.add.at(new_vs, inv, vs)
        counts = np.bincount(inv)
        new_vs /= counts[:, None]
        new_fs = inv[fs].astype(np.uint32).copy()  # ensure contiguous
        return new_vs, new_fs
