import os, ctypes
import numpy as np
import pyglet.gl as gl
import one.collider.gpu_collision_batch as ocgcb


class GPUBatchDetector:
    def __init__(self):
        self._program = None
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        unified_src = self._load_shader_file(os.path.join(shader_dir, 'twophase_batch.comp'))
        self._program = self._compile_shader_manual(unified_src, 'unified')

    def detect_collision_batch(self, batch, eps=1e-9):
        # update GPU buffers
        # batch.ensure_gpu_ready() # do nothing if already ready
        batch.sync_transforms()
        batch.clear_counter()
        # dispatch tritri
        total = int(batch.pair_prefix[-1])
        if total == 0:
            return None
        num_groups = (total + 255) // 256
        gl.glUseProgram(self._program)
        self._set_uniforms(eps, batch._max_points)
        gl.glDispatchCompute(num_groups, 1, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
        gl.glUseProgram(0)
        # read back results
        count = batch.read_counter()
        if count == 0:
            return None
        count = min(count, batch._max_points)
        points4 = batch.read_points(count)
        points = points4[:, :3]
        pair_ids = points4[:, 3].astype(np.uint32, copy=False)
        return points, pair_ids

    def _set_uniforms(self, eps, max_points):
        loc_eps = gl.glGetUniformLocation(self._program, b'u_eps')
        loc_max = gl.glGetUniformLocation(self._program, b'u_maxPoints')
        gl.glUniform1f(loc_eps, float(eps))
        gl.glUniform1ui(loc_max, int(max_points))

    def _compile_shader_manual(self, source, name):
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
        with open(filepath, 'r') as f:
            return f.read()

def create_detector():
    return GPUBatchDetector()

def build_batch(items, pairs):
    return ocgcb.GPUCollisionBatch(items, pairs)
