import ctypes
import numpy as np
import pyglet.gl as gl


class DeviceBufferBase:

    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ebo = 0
        self.count = 0


class MeshBuffer(DeviceBufferBase):

    def __init__(self, verts, faces, vert_normals):
        super().__init__()
        self.instance_tf_vbo = 0
        self.instance_rgba_vbo = 0
        self.instance_count = 0
        # Cache for instance data
        self._cached_tf = None
        self._cached_rgba = None
        self._tf_buffer_size = 0
        self._rgba_buffer_size = 0
        self._build(verts, faces, vert_normals)

    def update_instances(self, tf_arr, rgba_array=None):
        """
        Update instance transform and color data with intelligent caching.
        Only uploads GPU data if it has changed or buffer needs resizing.
        """
        self.instance_count = len(tf_arr)
        gl.glBindVertexArray(self.vao)
        # Update transform buffer
        self.instance_tf_vbo, self._cached_tf, self._tf_buffer_size, _ = \
            self._update_instance_buffer(
                tf_arr, self.instance_tf_vbo,
                self._cached_tf, self._tf_buffer_size)
        # Setup transform vertex attributes
        stride = 16 * 4
        for i in range(4):
            loc = 2 + i  # location = 2,3,4,5
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc, 4, gl.GL_FLOAT, False, stride,
                ctypes.c_void_p(i * 16))
            gl.glVertexAttribDivisor(loc, 1)
        # Update RGBA buffer if provided
        if rgba_array is not None:
            self.instance_rgba_vbo, self._cached_rgba, self._rgba_buffer_size, _ = \
                self._update_instance_buffer(
                    rgba_array, self.instance_rgba_vbo,
                    self._cached_rgba, self._rgba_buffer_size)
            # Setup RGBA vertex attributes
            gl.glEnableVertexAttribArray(6)  # location = 6
            gl.glVertexAttribPointer(6, 4, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
            gl.glVertexAttribDivisor(6, 1)
        gl.glBindVertexArray(0)

    def draw_instanced(self):
        if self.instance_count <= 0:
            return
        gl.glBindVertexArray(self.vao)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, self.count, gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0), self.instance_count)
        # gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self.count, self.instance_count)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBindVertexArray(0)

    def _update_instance_buffer(
            self, data, vbo_id, cached_data, buffer_size):
        data_changed = cached_data is None or not np.array_equal(data, cached_data)
        needs_realloc = data.nbytes > buffer_size
        # Create VBO if needed
        if vbo_id == 0:
            buf = (gl.GLuint * 1)()
            gl.glGenBuffers(1, buf)
            vbo_id = buf[0]
            needs_realloc = True
            data_changed = True
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id)
        # Reallocate with headroom if needed
        if needs_realloc:
            buffer_size = int(data.nbytes * 1.5)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, buffer_size,
                None, gl.GL_DYNAMIC_DRAW)
            data_changed = True
        # Upload data if changed
        if data_changed:
            gl.glBufferSubData(
                gl.GL_ARRAY_BUFFER, 0,
                data.nbytes, data.ctypes.data)
            cached_data = data.copy()
        return vbo_id, cached_data, buffer_size, data_changed

    def _build(self, verts, faces, vert_normals):
        array = np.hstack(
            [verts, vert_normals]).astype(np.float32)
        # create VAO (vertex array object), VBO and EBO will be bound to it
        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self.vao = vao[0]
        # VBO, vertex buffer object
        vbo = (gl.GLuint * 1)()
        gl.glGenBuffers(1, vbo)
        self.vbo = vbo[0]
        # bind VAO
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, array.nbytes,
            array.ctypes.data, gl.GL_STATIC_DRAW)
        stride = 6 * 4  # float32 * 6
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, False,
            stride, ctypes.c_void_p(0))
        # a_normal (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, False,
            stride, ctypes.c_void_p(12))
        # # EBO
        ebo = (gl.GLuint * 1)()
        gl.glGenBuffers(1, ebo)
        self.ebo = ebo[0]
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # indices = faces.astype(np.uint32).copy(order='C')
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes,
                        faces.ctypes.data, gl.GL_STATIC_DRAW)
        self.count = faces.size
        gl.glBindVertexArray(0)


class PointCloudBuffer(DeviceBufferBase):
    def __init__(self, verts, per_vert_rgbs):
        super().__init__()
        self._build(verts, per_vert_rgbs)

    def draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)

    def _build(self, verts, per_vert_rgbs):
        self.count = len(verts)
        # color
        array = np.hstack(
            [verts, per_vert_rgbs]).astype(np.float32)
        # create VAO (vertex array object), VBO and EBO will be bound to it
        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self.vao = vao[0]
        # VBO, vertex buffer object
        vbo = (gl.GLuint * 1)()
        gl.glGenBuffers(1, vbo)
        self.vbo = vbo[0]
        # bind VAO
        gl.glBindVertexArray(self.vao)
        # bind VBO buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, array.nbytes,
            array.ctypes.data, gl.GL_STATIC_DRAW)
        stride = 6 * 4  # float32 * 6
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, False,
            stride, ctypes.c_void_p(0))
        # a_color (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, False,
            stride, ctypes.c_void_p(12))
        # unbind VAO
        gl.glBindVertexArray(0)
