import os
import ctypes
import numpy as np
import pyglet.gl as gl
import one.collider.collider_base as ocb
import one.collider.cpu_simd as occs


class GPUAABBCollider(ocb.ColliderBase):
    """
    GPU OBB/SAT collider. Interface compatible with AABBCollider.
    """
    def __init__(self):
        super().__init__()
        self._local_centers = None      # (N,3)
        self._half_extents = None       # (N,3)
        self._transforms = None         # (N,4,4)
        self._num_pairs = 0
        # GPU resources
        self._program = None
        self._centers_ssbo = None
        self._extents_ssbo = None
        self._transforms_ssbo = None
        self._pairs_ssbo = None
        self._counter_ssbo = None

    def _post_compile(self):
        # Build local AABB for each object (union of all collisions)
        centers = []
        extents = []
        for obj in self._pair_items:
            tris = occs.cols_to_tris(obj.collisions)
            if tris is None:
                centers.append(np.zeros(3, dtype=np.float32))
                extents.append(np.zeros(3, dtype=np.float32))
                continue
            min_c, max_c = occs.compute_aabb(tris)
            center = (min_c + max_c) * 0.5
            half = (max_c - min_c) * 0.5
            centers.append(center)
            extents.append(half)
        centers = np.array(centers, dtype=np.float32)
        extents = np.array(extents, dtype=np.float32)
        self._local_centers = np.zeros((centers.shape[0], 4), dtype=np.float32)
        self._half_extents = np.zeros((extents.shape[0], 4), dtype=np.float32)
        self._local_centers[:, :3] = centers
        self._half_extents[:, :3] = extents
        self._num_pairs = 0 if self._check_pairs is None \
            else len(self._check_pairs)
        # Pre-allocate transforms buffer
        n_items = len(self._pair_items)
        self._transforms = np.zeros(
            (n_items, 4, 4), dtype=np.float32)
        # GPU init + upload
        self._lazy_init()
        self._upload_static_buffers()

    def is_collided(self, qs):
        if not self._compiled:
            raise RuntimeError('GPUAABBCollider must be compiled!')
        # FK update
        for actor, sl in self._actor_qs_slice.items():
            actor.fk(qs[sl])
        if self._num_pairs == 0:
            return False
        # Update transforms
        for i, obj in enumerate(self._pair_items):
            self._transforms[i] = obj.tf
        # Upload transforms
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._transforms_ssbo)
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0,
                           self._transforms.nbytes,
                           self._transforms.ctypes.data_as(ctypes.c_void_p))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        # Clear counter
        zero = np.zeros(1, dtype=np.uint32)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._counter_ssbo)
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0,
                           zero.nbytes, zero.ctypes.data_as(ctypes.c_void_p))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        # Dispatch
        gl.glUseProgram(self._program)
        num_groups = (self._num_pairs + 255) // 256
        gl.glDispatchCompute(num_groups, 1, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
        gl.glUseProgram(0)
        # Read counter
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._counter_ssbo)
        ptr = gl.glMapBufferRange(gl.GL_SHADER_STORAGE_BUFFER, 0, 4, gl.GL_MAP_READ_BIT)
        result = np.zeros(1, dtype=np.uint32)
        ctypes.memmove(result.ctypes.data_as(ctypes.c_void_p), ptr, 4)
        gl.glUnmapBuffer(gl.GL_SHADER_STORAGE_BUFFER)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return bool(result[0] > 0)

    # helpers
    def _lazy_init(self):
        if self._program is not None:
            return
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        shader_file = os.path.join(shader_dir, 'aabb.comp')
        with open(shader_file, 'r') as f:
            unified_src = f.read()
        self._program = self._compile_shader(unified_src)

    def _upload_static_buffers(self):
        # centers
        self._centers_ssbo = self._create_ssbo(
            self._local_centers)
        # extents
        self._extents_ssbo = self._create_ssbo(
            self._half_extents)
        # pairs
        pairs = self._check_pairs.astype(np.uint32, copy=False)
        self._pairs_ssbo = self._create_ssbo(pairs)
        # transforms (dynamic)
        self._transforms_ssbo = self._create_ssbo(
            self._transforms)
        # counter
        counter = np.zeros(1, dtype=np.uint32)
        self._counter_ssbo = self._create_ssbo(counter)
        # Bind to slots
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            0, self._centers_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            1, self._extents_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            2, self._transforms_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            3, self._pairs_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            4, self._counter_ssbo)

    def _create_ssbo(self, data):
        ssbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(ssbo))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, data.nbytes,
                        data.ctypes.data_as(ctypes.c_void_p),
                        gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return ssbo.value

    def _compile_shader(self, source):
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        src_bytes = source.encode('utf-8')
        src_ptr = ctypes.cast(ctypes.c_char_p(src_bytes),
                              ctypes.POINTER(ctypes.c_char))
        src_len = ctypes.c_int(len(src_bytes))
        gl.glShaderSource(
            shader, 1, ctypes.byref(src_ptr), src_len)
        gl.glCompileShader(shader)
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS,
                         ctypes.byref(status))
        if not status.value:
            log_len = gl.GLint()
            gl.glGetShaderiv(
                shader, gl.GL_INFO_LOG_LENGTH,
                ctypes.byref(log_len))
            log = (gl.GLchar * log_len.value)()
            gl.glGetShaderInfoLog(
                shader, log_len.value, None, log)
            gl.glDeleteShader(shader)
            raise RuntimeError(log.value.decode('utf-8'))
        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        link_status = gl.GLint()
        gl.glGetProgramiv(program, gl.GL_LINK_STATUS,
                          ctypes.byref(link_status))
        if not link_status.value:
            log_len = gl.GLint()
            gl.glGetProgramiv(program, gl.GL_INFO_LOG_LENGTH,
                              ctypes.byref(log_len))
            log = (gl.GLchar * log_len.value)()
            gl.glGetProgramInfoLog(
                program, log_len.value, None, log)
            gl.glDeleteProgram(program)
            raise RuntimeError(log.value.decode('utf-8'))
        gl.glDeleteShader(shader)
        return program

    def _shader_source(self):
        return GPU_AABB_SHADER
