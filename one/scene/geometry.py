import numpy as np
import one.utils.constant as const


class GeometryBase:

    def __init__(self, verts, faces=None):
        self.verts = np.asarray(verts, dtype=np.float32)
        self.faces = None if faces is None else np.asarray(faces, dtype=np.uint32)
        self.device_buffer = None


class Mesh(GeometryBase):
    def __init__(self,
                 verts,
                 faces,
                 rgb=None,
                 face_normals=None,
                 vert_normals=None,
                 alpha=1.0):
        super().__init__(verts=verts, faces=faces)
        self.face_normals = face_normals
        self.vert_normals = vert_normals
        if self.faces is not None and self.face_normals is None:
            self.compute_face_normals()
        self.rgbs = self._ensure_rgbs(rgb, len(self.verts))
        self.alpha = alpha

    def compute_face_normals(self):
        if self.faces is None:
            print("Warning: Cannot compute face normals without faces.")
            return None
        f = self.faces
        v1 = self.verts[f[:, 1]] - self.verts[f[:, 0]]
        v2 = self.verts[f[:, 2]] - self.verts[f[:, 0]]
        fnormals = np.cross(v1, v2)
        fnormals /= np.linalg.norm(fnormals, axis=1, keepdims=True)
        self.face_normals = fnormals.astype(np.float32)
        return self.face_normals

    def compute_vert_normals(self):
        if self.faces is None:
            print("Warning: Cannot compute vertex normals without faces.")
            return None
        if self.face_normals is None:
            self.compute_face_normals()
        vnormals = np.zeros_like(self.verts, dtype=np.float32)
        np.add.at(vnormals, self.faces[:, 0], self.face_normals)
        np.add.at(vnormals, self.faces[:, 1], self.face_normals)
        np.add.at(vnormals, self.faces[:, 2], self.face_normals)
        vnormals /= np.linalg.norm(vnormals, axis=1, keepdims=True)
        self.vert_normals = vnormals.astype(np.float32)
        return self.vert_normals

    def set_rgba(self, rgb, alpha=1.0):
        self.rgb = self._ensure_rgbs(rgb, len(self.verts))
        self.alpha = alpha

    def _ensure_rgbs(self, c, n_verts):
        if c is None:
            return const.BasicColor.DEFAULT
        c = np.asarray(c, dtype=np.float32)
        # single RGBA
        if c.ndim == 1 and c.shape == (3,):
            return c
        # per-vertex RGBA
        if c.ndim == 2 and c.shape == (n_verts, 3):
            return c
        raise ValueError("Invalid color format. Expected (3,) or (N,3)")


class PointCloud(GeometryBase):
    def __init__(self, verts, colors=None):
        super().__init__(verts, faces=None)
        self.colors = colors


if __name__ == '__main__':
    import one.scene.loader as loader

    verts, faces = loader.load_stl("bunnysim.stl")
    geom = GeometryBase(verts=verts, faces=faces)
    print("Face normals:", geom.compute_face_normals())
    print("Is point cloud:", geom.is_point_cloud)
    print("RGBs:", geom.rgbs)
