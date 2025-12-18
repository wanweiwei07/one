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
            self.verts = rm.asarray(verts, dtype=rm.float32)
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
        raw_fns = rm.cross(v1, v2).astype(rm.float32)
        # vert normals
        vns = rm.zeros_like(self.verts)
        rm.add.at(vns, self.faces[:, 0], raw_fns)
        rm.add.at(vns, self.faces[:, 1], raw_fns)
        rm.add.at(vns, self.faces[:, 2], raw_fns)
        _, vns = rm.unit_vec(vns)
        return vns

    def _merge_vertices_and_faces(self, verts, faces, tol=1e-6):
        q = rm.round(verts / tol).astype(rm.int64)
        unique_q, inv = rm.unique(q, axis=0, return_inverse=True)
        verts_new = rm.zeros((len(unique_q), 3), dtype=verts.dtype)
        rm.add.at(verts_new, inv, verts)
        counts = rm.bincount(inv)
        verts_new /= counts[:, None]
        faces_new = inv[faces].astype(rm.uint32).copy() # ensure contiguous
        return verts_new, faces_new
