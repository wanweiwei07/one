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
        self.instance_tfmat_vbo = 0
        self.instance_rgba_vbo = 0
        self.instance_count = 0
        self._build(verts, faces, vert_normals)

    def update_instances(self, tf_mat_array, rgba_array=None):
        self.instance_count = len(tf_mat_array)
        gl.glBindVertexArray(self.vao)
        # instance tfmat VBO
        if self.instance_tfmat_vbo == 0:
            buf = (gl.GLuint * 1)()
            gl.glGenBuffers(1, buf)
            self.instance_tfmat_vbo = buf[0]
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_tfmat_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            tf_mat_array.nbytes,
            tf_mat_array.ctypes.data,
            gl.GL_DYNAMIC_DRAW,
        )
        stride = 16 * 4
        for i in range(4):
            loc = 2 + i  # location = 2,3,4,5
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc, 4, gl.GL_FLOAT, False, stride, ctypes.c_void_p(i * 16)
            )
            gl.glVertexAttribDivisor(loc, 1)
        if rgba_array is not None:
            # instance color VBO
            if self.instance_rgba_vbo == 0:
                buf = (gl.GLuint * 1)()
                gl.glGenBuffers(1, buf)
                self.instance_rgba_vbo = buf[0]
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_rgba_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, rgba_array.nbytes,
                            rgba_array.ctypes.data, gl.GL_DYNAMIC_DRAW)
            gl.glEnableVertexAttribArray(6)  # location = 6
            gl.glVertexAttribPointer(
                6, 4, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0)  # index  # vec4
            )
            gl.glVertexAttribDivisor(6, 1)
        gl.glBindVertexArray(0)

    def draw_instanced(self):
        if self.instance_count <= 0:
            return
        gl.glBindVertexArray(self.vao)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.count, gl.GL_UNSIGNED_INT,
                                   ctypes.c_void_p(0), self.instance_count)
        # gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self.count, self.instance_count)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBindVertexArray(0)

    def _build(self, verts, faces, vert_normals):
        array = np.hstack([verts, vert_normals]).astype(np.float32)
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
        gl.glBufferData(gl.GL_ARRAY_BUFFER, array.nbytes, array.ctypes.data, gl.GL_STATIC_DRAW)
        stride = 6 * 4  # float32 * 6
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        # a_normal (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
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
        array = np.hstack([verts, per_vert_rgbs]).astype(np.float32)
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
            gl.GL_ARRAY_BUFFER, array.nbytes, array.ctypes.data, gl.GL_STATIC_DRAW
        )
        stride = 6 * 4  # float32 * 6
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        # a_color (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        # unbind VAO
        gl.glBindVertexArray(0)
