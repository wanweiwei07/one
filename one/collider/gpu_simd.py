import os
import ctypes
import numpy as np
import pyglet.gl as gl


class GPUCollider:
    """
    GPU-accelerated triangle-triangle collision detection.
    Uses vao and vbo buffers already on GPU for input meshes.
    """

    def __init__(self):
        self._program = None
        self._initialized = False

    def detect_collision_single(self, col_a, tf_a, col_b, tf_b,
                                eps=1e-9, max_points=200):
        """
        detect collision between two collision shapes
        :param col_a, col_b: collision shape objects
        :param tf_a, tf_b: world transform matrices (4x4)
        :param eps: numerical tolerance (default 1e-9)
        :param max_points: max collision points to return (default 200)
        :return: (N,3) collision points or None
        """
        tf_a = tf_a @ col_a.tf
        tf_b = tf_b @ col_b.tf
        geom_a = col_a.geom
        geom_b = col_b.geom
        device_a = geom_a.get_device_buffer()
        device_b = geom_b.get_device_buffer()
        if device_a is None or device_b is None:
            raise RuntimeError("Geometry device buffers not available (OpenGL context required)")
        vbo_a = device_a.vbo
        ebo_a = device_a.ebo
        vbo_b = device_b.vbo
        ebo_b = device_b.ebo
        num_tris_a = len(geom_a.fs)
        num_tris_b = len(geom_b.fs)
        return self._run_collision(vbo_a, ebo_a, num_tris_a, tf_a,
                                   vbo_b, ebo_b, num_tris_b, tf_b,
                                   eps, max_points)

    def _run_collision(self, vbo_a, ebo_a, num_tris_a, tf_a,
                       vbo_b, ebo_b, num_tris_b, tf_b, eps, max_points):
        """
        execute single-pass GPU collision detection
        :param vbo_a, ebo_a: vertex/element buffer handles for mesh A
        :param num_tris_a: triangle count for mesh A
        :param tf_a: world transform matrix (4x4) for mesh A
        :param vbo_b, ebo_b: vertex/element buffer handles for mesh B
        :param num_tris_b: triangle count for mesh B
        :param tf_b: world transform matrix (4x4) for mesh B
        :param eps: numerical tolerance
        :param max_points: maximum collision points to return
        :return: (N,3) collision points or None
        """
        self._lazy_init()
        buffer_size = min(max_points, min(num_tris_a, num_tris_b))
        points = np.zeros((buffer_size, 4), dtype=np.float32)
        counter = np.zeros(1, dtype=np.uint32)
        points_ssbo = self._create_ssbo(points)
        counter_ssbo = self._create_ssbo(counter)
        try:
            gl.glUseProgram(self._program)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, vbo_a)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, ebo_a)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, vbo_b)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 3, ebo_b)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, points_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 5, counter_ssbo)
            self._set_uniforms(tf_a, tf_b, num_tris_a, num_tris_b, eps, buffer_size)
            total_pairs = num_tris_a * num_tris_b
            num_workgroups = (total_pairs + 255) // 256
            gl.glDispatchCompute(num_workgroups, 1, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
            gl.glUseProgram(0)
            result_count = self._read_ssbo(counter_ssbo, np.uint32, 1)[0]
            if result_count == 0:
                return None
            actual_count = min(result_count, buffer_size)
            if result_count > buffer_size:
                import sys
                print(f"WARNING: Collision points truncated: {result_count} detected, "
                      f"only {buffer_size} returned (max_points={max_points} limit). "
                      f"Increase max_points parameter if needed.", file=sys.stderr)
            result_points = self._read_ssbo(points_ssbo, np.float32, actual_count * 4)
            result_points = result_points.reshape(-1, 4)
            return result_points[:, :3]
        finally:
            self._delete_ssbo(points_ssbo)
            self._delete_ssbo(counter_ssbo)

    def _lazy_init(self):
        """initialize unified shader on first use (requires OpenGL context)"""
        if self._initialized:
            return
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        unified_src = self._load_shader_file(os.path.join(shader_dir, 'twophase_vbo.comp'))
        try:
            self._program = self._compile_shader_manual(unified_src, 'unified')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"Failed to compile compute shader:\n{e}\n{tb}")
        self._initialized = True

    def _compile_shader_manual(self, source, name):
        """
        compile compute shader without pyglet's uniform introspection
        :param source: GLSL source code string
        :param name: shader name for error messages
        :return: compiled OpenGL program handle
        """
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        src_bytes = source.encode('utf-8')
        src_ptr = ctypes.cast(ctypes.c_char_p(src_bytes), ctypes.POINTER(ctypes.c_char))
        src_len = ctypes.c_int(len(src_bytes))
        gl.glShaderSource(shader, 1, ctypes.byref(src_ptr), src_len)
        gl.glCompileShader(shader)
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            log_len = gl.GLint()
            gl.glGetShaderiv(shader, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
            log = (gl.GLchar * log_len.value)()
            gl.glGetShaderInfoLog(shader, log_len.value, None, log)
            gl.glDeleteShader(shader)
            raise RuntimeError(f"Shader compilation failed ({name}):\n{log.value.decode('utf-8')}")
        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        link_status = gl.GLint()
        gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(link_status))
        if not link_status.value:
            log_len = gl.GLint()
            gl.glGetProgramiv(program, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
            log = (gl.GLchar * log_len.value)()
            gl.glGetProgramInfoLog(program, log_len.value, None, log)
            gl.glDeleteShader(shader)
            gl.glDeleteProgram(program)
            raise RuntimeError(f"Shader linking failed ({name}):\n{log.value.decode('utf-8')}")
        gl.glDeleteShader(shader)
        return program

    def _load_shader_file(self, filepath):
        """load shader source from file"""
        with open(filepath, 'r') as f:
            return f.read()

    def _create_ssbo(self, data):
        """
        create Shader Storage Buffer Object from numpy array
        :param data: numpy array to upload
        :return: SSBO handle (GLuint)
        """
        ssbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(ssbo))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, data.nbytes,
                        data.ctypes.data_as(ctypes.c_void_p), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return ssbo.value

    def _read_ssbo(self, ssbo, dtype, count):
        """
        read data from SSBO back to CPU
        :param ssbo: SSBO handle
        :param dtype: numpy dtype
        :param count: number of elements
        :return: numpy array
        """
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

    def _delete_ssbo(self, ssbo):
        """delete SSBO and free GPU memory"""
        ssbo_id = gl.GLuint(ssbo)
        gl.glDeleteBuffers(1, ctypes.byref(ssbo_id))

    def _set_uniforms(self, tf_a, tf_b, num_tris_a, num_tris_b, eps, max_points):
        """set shader uniform variables"""
        loc_tf_a = gl.glGetUniformLocation(self._program, b'u_transformA')
        loc_tf_b = gl.glGetUniformLocation(self._program, b'u_transformB')
        loc_ntris_a = gl.glGetUniformLocation(self._program, b'u_numTrianglesA')
        loc_ntris_b = gl.glGetUniformLocation(self._program, b'u_numTrianglesB')
        loc_eps = gl.glGetUniformLocation(self._program, b'u_eps')
        loc_max = gl.glGetUniformLocation(self._program, b'u_maxPoints')
        tf_a_ptr = tf_a.astype(np.float32).ctypes.data_as(ctypes.POINTER(gl.GLfloat))
        tf_b_ptr = tf_b.astype(np.float32).ctypes.data_as(ctypes.POINTER(gl.GLfloat))
        gl.glUniformMatrix4fv(loc_tf_a, 1, gl.GL_TRUE, tf_a_ptr)
        gl.glUniformMatrix4fv(loc_tf_b, 1, gl.GL_TRUE, tf_b_ptr)
        gl.glUniform1ui(loc_ntris_a, num_tris_a)
        gl.glUniform1ui(loc_ntris_b, num_tris_b)
        gl.glUniform1f(loc_eps, eps)
        gl.glUniform1ui(loc_max, max_points)


_gpu_collider = None


def _get_gpu_collider():
    """get or create the global GPU collider singleton"""
    global _gpu_collider
    if _gpu_collider is None:
        _gpu_collider = GPUCollider()
    return _gpu_collider


def detect_collision(col_a, tf_a, col_b, tf_b,
                     eps=1e-9, max_points=1000):
    """
    detect collision between two collision shapes using GPU
    NOTE: broad aabb check should be done prior to calling this function for efficiency
    :param col_a, col_b: CollisionShape instances
    :param tf_a, tf_b: (4,4) world transform matrices
    :param eps: numerical tolerance (default 1e-9)
    :param max_points: max collision points to return (default 200)
    :return: (K,3) collision points or None
    """
    collider = _get_gpu_collider()
    return collider.detect_collision_single(
        col_a, tf_a, col_b, tf_b, eps, max_points)