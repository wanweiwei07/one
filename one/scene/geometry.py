import numpy as np


class Geometry:

    def __init__(self,
                 verts,
                 faces,
                 face_normals=None,
                 vert_normals=None,
                 per_vert_rgbs=None):
        self.verts = np.asarray(verts, dtype=np.float32)
        self.faces = None if faces is None else np.asarray(faces, dtype=np.uint32)
        self.face_normals = face_normals
        self.vert_normals = vert_normals
        self.per_vert_rgbs = per_vert_rgbs
        if self.faces is not None and self.face_normals is None:
            self.compute_face_normals()
        self.device_buffer = None

    def compute_face_normals(self):
        if self.faces is None:
            print("Warning: Cannot compute face normals without faces.")
            return None
        f = self.faces
        v1 = self.verts[f[:, 1]] - self.verts[f[:, 0]]
        v2 = self.verts[f[:, 2]] - self.verts[f[:, 0]]
        fns = np.cross(v1, v2)
        fns /= np.linalg.norm(fns, axis=1, keepdims=True)
        self.face_normals = fns.astype(np.float32)
        return self.face_normals

    def compute_vert_normals(self):
        if self.faces is None:
            print("Warning: Cannot compute vertex normals without faces.")
            return None
        if self.face_normals is None:
            self.compute_face_normals()
        vns = np.zeros_like(self.verts, dtype=np.float32)
        np.add.at(vns, self.faces[:, 0], self.face_normals)
        np.add.at(vns, self.faces[:, 1], self.face_normals)
        np.add.at(vns, self.faces[:, 2], self.face_normals)
        vns /= np.linalg.norm(vns, axis=1, keepdims=True)
        self.vert_normals = vns.astype(np.float32)
        return self.vert_normals
