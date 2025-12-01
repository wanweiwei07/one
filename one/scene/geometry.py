import numpy as np
import wrs.utils.constant as const

class Geometry:
    def __init__(self,
                 verts=None,
                 faces=None,
                 face_normals=None,
                 rgbs=None,
                 alpha=1.0):
        self.verts = verts
        self.faces = faces
        self.face_normals = face_normals
        if faces is not None and face_normals is None:
            self.compute_face_normals()
        self.rgbs = self._ensure_rgbs(rgbs, len(self.verts))
        self.alpha = alpha

    def compute_face_normals(self):
        if self.faces is None:
            return None
        f = self.faces
        v1 = self.verts[f[:, 1]] - self.verts[f[:, 0]]
        v2 = self.verts[f[:, 2]] - self.verts[f[:, 0]]
        normals = np.cross(v1, v2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.face_normals = normals.astype(np.float32)
        return self.face_normals

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

    @property
    def is_point_cloud(self):
        return self.faces is None


if __name__ == '__main__':
    from wrs import loader

    verts, faces = loader.load_stl("bunnysim.stl")
    geom = Geometry(verts=verts, faces=faces)
    print("Face normals:", geom.compute_face_normals())
    print("Is point cloud:", geom.is_point_cloud)
    print("RGBs:", geom.rgbs)
