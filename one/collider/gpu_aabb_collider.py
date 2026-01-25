import os, ctypes, time
import numpy as np, pyglet.gl as gl
import one.collider.collider_base as ocb
import one.collider.cpu_simd as occs


class GPUAABBCollider(ocb.ColliderBase):
    """
    GPU OBB/SAT collider with full async pipeline optimization.
    """

    def __init__(self):
        super().__init__()
        self._local_centers = None
        self._half_extents = None
        self._transforms = None
        self._num_pairs = 0
        # GPU resources
        self._program = None
        self._centers_ssbo = None
        self._extents_ssbo = None
        self._pairs_ssbo = None
        # Triple buffering with persistent mapping
        self._num_buffers = 3
        self._current_frame = 0
        self._transforms_ssbos = []
        self._counter_ssbos = []
        self._transforms_ptrs = []  # Persistent mapped pointers
        self._counter_ptrs = []
        self._fences = [None] * self._num_buffers
        # Performance profiling
        self._enable_profiling = True
        self._profile_stats = {
            'fk': [], 'update_tf': [], 'fence_wait': [],
            'memcpy': [], 'dispatch': [], 'read': [], 'total': []
        }
        self._call_count = 0

    def _post_compile(self):
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
        self._num_pairs = 0 if self._check_pairs is None else len(self._check_pairs)
        # pre-allocate transform buffer
        n_items = len(self._pair_items)
        self._transforms = np.zeros((n_items, 4, 4), dtype=np.float32)
        # initialize GPU resources
        self._lazy_init()
        self._upload_static_buffers()

    def _upload_static_buffers(self):
        # static data
        self._centers_ssbo = self._create_ssbo(self._local_centers, persistent=False)
        self._extents_ssbo = self._create_ssbo(self._half_extents, persistent=False)
        pairs = self._check_pairs.astype(np.uint32, copy=False)
        self._pairs_ssbo = self._create_ssbo(pairs, persistent=False)
        # clear lists for triple buffering
        self._transforms_ssbos = []
        self._counter_ssbos = []
        self._transforms_ptrs = []
        self._counter_ptrs = []
        # create triple buffers
        for _ in range(self._num_buffers):
            # Transforms buffer with persistent mapping (CPU-GPU)
            ssbo, ptr = self._create_persistent_ssbo(self._transforms.nbytes)
            self._transforms_ssbos.append(ssbo)
            self._transforms_ptrs.append(ptr)
            # Counter buffer with persistent mapping (GPU-CPU)
            ssbo, ptr = self._create_persistent_ssbo(4)  # 4 bytes for uint32
            self._counter_ssbos.append(ssbo)
            self._counter_ptrs.append(ptr)
        # bind static SSBOs
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self._centers_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self._extents_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 3, self._pairs_ssbo)

    def is_collided(self, qs):
        if not self._compiled:
            raise RuntimeError('GPUAABBCollider must be compiled!')
        t_start = time.perf_counter()
        # fk
        t0 = time.perf_counter()
        for actor, sl in self._actor_qs_slice.items():
            actor.fk(qs[sl])
        t1 = time.perf_counter()
        # early out if no pairs
        if self._num_pairs == 0:
            return False
        # update transforms
        t2 = time.perf_counter()
        for i, item in enumerate(self._pair_items):
            self._transforms[i] = item.tf
        t3 = time.perf_counter()
        # get current frame resources
        frame_idx = self._current_frame
        transforms_ssbo = self._transforms_ssbos[frame_idx]
        counter_ssbo = self._counter_ssbos[frame_idx]
        transforms_ptr = self._transforms_ptrs[frame_idx]
        counter_ptr = self._counter_ptrs[frame_idx]
        # wait for previous frame to complete
        t4 = time.perf_counter()
        if self._fences[frame_idx] is not None:
            result = gl.glClientWaitSync(
                self._fences[frame_idx],
                0,  # no flush
                0)  # check immediately
            # wait at most 100us if not signaled
            # TODO: potential case of race condition here
            # TODO: better strategy to handle long GPU stalls
            if result == gl.GL_TIMEOUT_EXPIRED:
                gl.glClientWaitSync(
                    self._fences[frame_idx],
                    gl.GL_SYNC_FLUSH_COMMANDS_BIT,
                    100000)  # 100us
            gl.glDeleteSync(self._fences[frame_idx])
            self._fences[frame_idx] = None
        t5 = time.perf_counter()
        # upload transforms and reset counter
        t6 = time.perf_counter()
        ctypes.memmove(transforms_ptr,
                       self._transforms.ctypes.data_as(ctypes.c_void_p),
                       self._transforms.nbytes)
        ctypes.memset(counter_ptr, 0, 4)
        t7 = time.perf_counter()
        # bind dynamic SSBOs
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, transforms_ssbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, counter_ssbo)
        # dispatch compute shader
        t8 = time.perf_counter()
        gl.glUseProgram(self._program)
        num_groups = (self._num_pairs + 255) // 256
        gl.glDispatchCompute(num_groups, 1, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
        gl.glUseProgram(0)
        t9 = time.perf_counter()
        # insert fence for this frame
        self._fences[frame_idx] = gl.glFenceSync(
            gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        # read results
        t10 = time.perf_counter()
        if self._current_frame < self._num_buffers - 1:
            self._current_frame += 1
            return False
        read_idx = (frame_idx + 1) % self._num_buffers
        result_ptr = self._counter_ptrs[read_idx]
        result = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_uint32))[0]
        t11 = time.perf_counter()
        self._current_frame = (self._current_frame + 1) % self._num_buffers
        # performance logging
        t_end = time.perf_counter()
        if self._enable_profiling:
            self._profile_stats['fk'].append((t1 - t0) * 1000)
            self._profile_stats['update_tf'].append((t3 - t2) * 1000)
            self._profile_stats['fence_wait'].append((t5 - t4) * 1000)
            self._profile_stats['memcpy'].append((t7 - t6) * 1000)
            self._profile_stats['dispatch'].append((t9 - t8) * 1000)
            self._profile_stats['read'].append((t11 - t10) * 1000)
            self._profile_stats['total'].append((t_end - t_start) * 1000)
            self._call_count += 1
            # print stats every 100 calls
            if self._call_count % 100 == 0:
                self._print_profile_stats()
        return bool(result > 0)

    def _print_profile_stats(self):
        print("\n" + "=" * 70)
        print(f"Performance Stats (last 100 calls, {self._num_pairs} collision pairs)")
        print("=" * 70)
        for name, times in self._profile_stats.items():
            if not times:
                continue
            avg = np.mean(times[-100:])
            std = np.std(times[-100:])
            min_t = np.min(times[-100:])
            max_t = np.max(times[-100:])
            # percentage of total time
            total_avg = np.mean(self._profile_stats['total'][-100:])
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            print(f"{name:12s}: {avg:6.3f}ms ± {std:5.3f}ms  "
                  f"[{min_t:6.3f} - {max_t:6.3f}]  ({pct:5.1f}%)")
        print("=" * 70)
        # trim stats to last 1000 entries
        for key in self._profile_stats:
            self._profile_stats[key] = self._profile_stats[key][-1000:]

    def get_profile_summary(self):
        if not self._profile_stats['total']:
            return "No profiling data available"
        summary = {}
        for name, times in self._profile_stats.items():
            if times:
                summary[name] = {
                    'avg_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times)}
        return summary

    def _lazy_init(self):
        if self._program is not None:
            return
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        shader_file = os.path.join(shader_dir, 'aabb.comp')
        with open(shader_file, 'r') as f:
            unified_src = f.read()
        self._program = self._compile_shader(unified_src)

    def _create_ssbo(self, data, persistent=False):
        """ssbo for static data"""
        ssbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(ssbo))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, data.nbytes,
                        data.ctypes.data_as(ctypes.c_void_p),
                        gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return ssbo.value

    def _create_persistent_ssbo(self, size_bytes):
        """ssbo for dynamic data with persistent mapping"""
        ssbo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(ssbo))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
        flags = (gl.GL_MAP_WRITE_BIT |
                 gl.GL_MAP_READ_BIT |
                 gl.GL_MAP_PERSISTENT_BIT |
                 gl.GL_MAP_COHERENT_BIT)
        gl.glBufferStorage(
            gl.GL_SHADER_STORAGE_BUFFER,
            size_bytes,
            None,  # uninitialized
            flags)
        # persistent map
        ptr = gl.glMapBufferRange(
            gl.GL_SHADER_STORAGE_BUFFER,
            0,
            size_bytes,
            flags)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return ssbo.value, ptr

    def _compile_shader(self, source):
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        src_bytes = source.encode('utf-8')
        src_ptr = ctypes.cast(ctypes.c_char_p(src_bytes),
                              ctypes.POINTER(ctypes.c_char))
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
            raise RuntimeError(log.value.decode('utf-8'))
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
            gl.glDeleteProgram(program)
            raise RuntimeError(log.value.decode('utf-8'))
        gl.glDeleteShader(shader)
        return program
