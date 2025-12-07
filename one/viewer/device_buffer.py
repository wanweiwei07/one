import numpy as np
import pyglet.gl as gl
import ctypes


class DeviceBuffer:
    def __init__(self, geometry):
        self._geom = geometry
        self.vao = 0
        self.vbo = 0
        self.count = 0
        if self._geom.faces is None:
            self.mode = gl.GL_POINTS
            self._build_pcd()
        else:
            self.mode = gl.GL_TRIANGLES
            self._build_mesh()

    def draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(self.mode, 0, self.count)
        gl.glBindVertexArray(0)

    # def update_vertices(self, verts):
    #     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
    #     gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

    def _build_mesh(self):
        verts_src = self._geom.verts
        if self._geom.faces is None:
            verts = verts_src
        else:
            verts = verts_src[self._geom.faces].reshape(-1, 3)
        self.count = len(verts)
        # normals (flat shading: use face normals)
        if self._geom.faces is None:
            normals = np.zeros_like(verts)
        elif self._geom.face_normals is not None:
            normals = np.repeat(self._geom.face_normals, 3, axis=0)
        else:
            raise Exception('No face normals')
        array = np.hstack([verts, normals]).astype(np.float32)
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
        stride = 6 * 4  # float32 * 9
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        # a_normal (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        # unbind VAO
        gl.glBindVertexArray(0)

    def _build_pcd(self):
        verts = self._geom.verts
        self.count = len(verts)
        # color
        rgbs = self._geom.per_vert_rgbs
        array = np.hstack([verts, rgbs]).astype(np.float32)
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
        gl.glBufferData(gl.GL_ARRAY_BUFFER, array.nbytes, array.ctypes.data, gl.GL_STATIC_DRAW)
        stride = 6 * 4  # float32 * 9
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        # a_color (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        # unbind VAO
        gl.glBindVertexArray(0)