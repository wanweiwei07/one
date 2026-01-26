import ctypes
import numpy as np, pyglet.gl as gl
from one.collider.collision_batch import CollisionBatch


class GPUCollisionBatch(CollisionBatch):
    def __init__(self, items, pairs=None):
        super().__init__(items, pairs)
        self.pair_prefix = self._build_pair_prefix()
        # GPU resource state
        self._max_points = 200
        # extended vss and fss
        self.vss4 = None
        self.fss4 = None
        # SSBO handles
        self._verts_ssbo = None
        self._faces_ssbo = None
        self._geom_descs_ssbo = None
        self._pairs_ssbo = None
        self._prefix_ssbo = None
        self._transforms_ssbo = None
        self._points_ssbo = None
        self._counter_ssbo = None
        # ensure ready
        self._ensure_gpu_ready()

    def update_transforms_ssbo(self):
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._transforms_ssbo)
        gl.glBufferSubData(
            gl.GL_SHADER_STORAGE_BUFFER, 0,
            self.tfs.nbytes,
            self.tfs.ctypes.data_as(ctypes.c_void_p))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)

    def sync_transforms(self):
        super().update_transforms()
        self.update_transforms_ssbo()

    def clear_counter(self):
        zero = np.zeros(1, dtype=np.uint32)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._counter_ssbo)
        gl.glBufferSubData(
            gl.GL_SHADER_STORAGE_BUFFER, 0,
            zero.nbytes,
            zero.ctypes.data_as(ctypes.c_void_p))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)

    def read_counter(self):
        return int(self._read_ssbo(self._counter_ssbo, np.uint32, 1)[0])

    def read_points(self, count):
        count = min(count, self._max_points)
        data = self._read_ssbo(self._points_ssbo, np.float32, count * 4)
        return data.reshape(-1, 4)

    def release(self):
        for ssbo in (
                self._verts_ssbo, self._faces_ssbo, self._geom_descs_ssbo,
                self._pairs_ssbo, self._prefix_ssbo, self._transforms_ssbo,
                self._points_ssbo, self._counter_ssbo):
            if ssbo is None:
                continue
            ssbo_id = gl.GLuint(ssbo)
            gl.glDeleteBuffers(1, ctypes.byref(ssbo_id))
        self._verts_ssbo = None
        self._faces_ssbo = None
        self._geom_descs_ssbo = None
        self._pairs_ssbo = None
        self._prefix_ssbo = None
        self._transforms_ssbo = None
        self._points_ssbo = None
        self._counter_ssbo = None
        self._gpu_ready = False

    def _ensure_gpu_ready(self):
        # Static buffers
        self.vss4 = np.ones((self.vss.shape[0], 4), dtype=np.float32)
        self.vss4[:, :3] = self.vss
        self.fss4 = np.ones((self.fss.shape[0], 4), dtype=np.uint32)
        self.fss4[:, :3] = self.fss
        self._verts_ssbo = self._create_ssbo(self.vss4.astype(np.float32, copy=False))
        self._faces_ssbo = self._create_ssbo(self.fss4.astype(np.uint32, copy=False))
        self._geom_descs_ssbo = self._create_ssbo(self.geom_descs.astype(np.uint32, copy=False))
        self._pairs_ssbo = self._create_ssbo(self.pairs.astype(np.uint32, copy=False))
        self._prefix_ssbo = self._create_ssbo(self.pair_prefix.astype(np.uint32, copy=False))
        # Dynamic buffers
        self._transforms_ssbo = self._create_ssbo(self.tfs.astype(np.float32, copy=False))
        self._points_ssbo = self._create_ssbo(np.zeros((self._max_points, 4), dtype=np.float32))
        self._counter_ssbo = self._create_ssbo(np.zeros(1, dtype=np.uint32))
        # Bind SSBOs to shader binding points (match shader)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self._verts_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self._faces_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, self._geom_descs_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 3, self._pairs_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, self._transforms_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 5, self._prefix_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 6, self._points_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 7, self._counter_ssbo)

    def _create_ssbo(self, data):
        ssbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(ssbo))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, data.nbytes,
                        data.ctypes.data_as(ctypes.c_void_p),
                        gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return ssbo.value

    def _read_ssbo(self, ssbo, dtype, count):
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
        buffer_ptr = gl.glMapBufferRange(gl.GL_SHADER_STORAGE_BUFFER, 0,
                                         count * np.dtype(dtype).itemsize,
                                         gl.GL_MAP_READ_BIT)
        result = np.zeros(count, dtype=dtype)
        ctypes.memmove(result.ctypes.data_as(ctypes.c_void_p), buffer_ptr,
                       count * np.dtype(dtype).itemsize)
        gl.glUnmapBuffer(gl.GL_SHADER_STORAGE_BUFFER)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return result

    def _build_pair_prefix(self):
        """index prefix for each pair's face count product"""
        counts = []
        for (i, j) in self.pairs:
            f_cnt_a = self.geom_descs[i][3]
            f_cnt_b = self.geom_descs[j][3]
            counts.append(np.uint32(f_cnt_a * f_cnt_b))
        prefix = np.zeros(len(counts) + 1, dtype=np.uint32)
        prefix[1:] = np.cumsum(np.array(
            counts, dtype=np.uint64)).astype(np.uint32)
        return prefix
