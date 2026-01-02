import numpy as np
import one.utils.math as rm
import one.viewer.device_buffer as dvb


class Geometry:

    def __init__(self,
                 verts,
                 faces,
                 per_vert_rgbs=None):
        if faces is not None:
            self.verts, self.faces = self._merge_vertices_and_faces(verts, faces)
            self.vert_normals = self._compute_vert_normals()
        else:
            self.verts = np.asarray(verts, dtype=np.float32)
            self.faces = None
            self.vert_normals = None
            self.per_vert_rgbs = per_vert_rgbs
        self._device_buffer = None

    def get_device_buffer(self):
        if self._device_buffer is None:
            if self.faces is None:
                self._device_buffer = dvb.PointCloudBuffer(self.verts, self.per_vert_rgbs)
            else:
                self._device_buffer = dvb.MeshBuffer(self.verts, self.faces,
                                                     self.vert_normals)
        return self._device_buffer

    def _compute_vert_normals(self):
        v1 = self.verts[self.faces[:, 1]] - self.verts[self.faces[:, 0]]
        v2 = self.verts[self.faces[:, 2]] - self.verts[self.faces[:, 0]]
        raw_fns = np.cross(v1, v2).astype(np.float32)
        _, unit_fns = rm.unit_vec(raw_fns)
        # vert normals
        vns = np.zeros_like(self.verts)
        np.add.at(vns, self.faces[:, 0], unit_fns)
        np.add.at(vns, self.faces[:, 1], unit_fns)
        np.add.at(vns, self.faces[:, 2], unit_fns)
        _, vns = rm.unit_vec(vns)
        return vns

    def _merge_vertices_and_faces(self, verts, faces, tol=1e-6):
        q = np.round(verts / tol).astype(np.int64)
        unique_q, inv = np.unique(q, axis=0, return_inverse=True)
        verts_new = np.zeros((len(unique_q), 3), dtype=verts.dtype)
        np.add.at(verts_new, inv, verts)
        counts = np.bincount(inv)
        verts_new /= counts[:, None]
        faces_new = inv[faces].astype(np.uint32).copy() # ensure contiguous
        return verts_new, faces_new
