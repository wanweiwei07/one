import numpy as np
import numpy as np


class MergedGeom:
    """Merged collision geom built from multiple collision shapes."""

    def __init__(self, vertices, faces):
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.uint32)
        self._aabb = None
        self._vertex_buffer = None
        self._gpu_vertices_ssbo = None
        self._gpu_faces_ssbo = None

    @classmethod
    def from_collisions(cls, collisions):
        """
        Build a merged_geom from collision shapes.
        :param collisions: list[CollisionShape]
        :return: MergedGeom or None when no triangles exist
        """
        if not collisions:
            return None
        all_verts = []
        all_faces = []
        vertex_offset = 0
        for col_shape in collisions:
            geom = col_shape.geom
            if geom is None or geom.fs is None or geom.vs is None:
                continue
            if geom.fs.size == 0 or geom.vs.size == 0:
                continue
            verts = (col_shape.rotmat @ geom.vs.T).T + col_shape.pos
            faces = np.asarray(geom.fs, dtype=np.uint32) + vertex_offset
            all_verts.append(verts)
            all_faces.append(faces)
            vertex_offset += len(verts)
        if not all_verts or not all_faces:
            return None
        merged_vertices = np.concatenate(all_verts, axis=0)
        merged_faces = np.concatenate(all_faces, axis=0)
        return cls(merged_vertices, merged_faces)

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        if self._vertices.size == 0:
            return None, None
        min_corner = self._vertices.min(axis=0)
        max_corner = self._vertices.max(axis=0)
        self._aabb = (min_corner, max_corner)
        return self._aabb

    def get_vertex_buffer(self):
        if self._vertex_buffer is None:
            if self._vertices.size == 0:
                self._vertex_buffer = np.zeros((0,), dtype=np.float32)
            else:
                self._vertex_buffer = self._vertices.reshape(-1).astype(np.float32, copy=False)
        return self._vertex_buffer

    def get_face_buffer(self):
        if self._faces is None or self._faces.size == 0:
            return np.zeros((0,), dtype=np.uint32)
        return np.asarray(self._faces, dtype=np.uint32).reshape(-1)


class MergedGeomBatch:
    """Batch data for GPU collision queries."""

    def __init__(self, merged_geoms, pair_indices):
        self._merged_geoms = list(merged_geoms)
        self._pair_indices = np.asarray(pair_indices, dtype=np.uint32)
        self._vertices = None
        self._faces = None
        self._geom_desc = None
        self._gpu_vertices_ssbo = None
        self._gpu_faces_ssbo = None
        self._gpu_geom_desc_ssbo = None
        self._gpu_pairs_ssbo = None

    @property
    def merged_geoms(self):
        return self._merged_geoms

    @property
    def pair_indices(self):
        return self._pair_indices

    @property
    def num_geoms(self):
        return len(self._merged_geoms)

    @property
    def num_pairs(self):
        return int(len(self._pair_indices))

    def get_vertices(self):
        self._build_buffers()
        return self._vertices

    def get_faces(self):
        self._build_buffers()
        return self._faces

    def get_geom_desc(self):
        self._build_buffers()
        return self._geom_desc

    def get_pairs(self):
        return self._pair_indices

    def _build_buffers(self):
        if self._vertices is not None:
            return
        vertex_chunks = []
        face_chunks = []
        geom_desc = []
        vertex_offset = 0
        face_offset = 0
        for geom in self._merged_geoms:
            if geom is None or geom.vertices is None or geom.fss is None:
                geom_desc.append([vertex_offset, face_offset, 0, 0])
                continue
            if geom.vertices.size == 0 or geom.fss.size == 0:
                geom_desc.append([vertex_offset, face_offset, 0, 0])
                continue
            vertex_buffer = geom.get_vertex_buffer()
            faces = geom.get_face_buffer() + vertex_offset
            vertex_count = len(geom.vertices)
            face_count = len(geom.fss)
            vertex_chunks.append(vertex_buffer)
            face_chunks.append(faces.astype(np.uint32, copy=False))
            geom_desc.append([vertex_offset, face_offset, vertex_count, face_count])
            vertex_offset += vertex_count
            face_offset += face_count
        if vertex_chunks:
            self._vertices = np.concatenate(vertex_chunks).astype(np.float32, copy=False)
        else:
            self._vertices = np.zeros((0,), dtype=np.float32)
        if face_chunks:
            self._faces = np.concatenate(face_chunks).astype(np.uint32, copy=False)
        else:
            self._faces = np.zeros((0,), dtype=np.uint32)
        self._geom_desc = np.asarray(geom_desc, dtype=np.uint32)
