import numpy as np
import pyglet.gl as gl
import ctypes

class DeviceBuffer:
    def __init__(self, geometry):
        self.geom = geometry
        self.vao = 0
        self.vbo = 0
        self.count = 0
        if self.geom.faces is None:
            self.mode = gl.GL_POINTS
        else:
            self.mode = gl.GL_TRIANGLES
        self._build()

    def draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(self.mode, 0, self.count)
        gl.glBindVertexArray(0)

    # def update_vertices(self, verts):
    #     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
    #     gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

    def _build(self):
        verts_src = self.geom.verts
        if self.geom.faces is None:
            verts = verts_src
        else:
            verts = verts_src[self.geom.faces].reshape(-1,3)
        self.count = len(verts)
        # normals (flat shading: use face normals)
        if self.geom.faces is None:
            normals = np.zeros_like(verts)
        elif self.geom.face_normals is not None:
            normals = np.repeat(self.geom.face_normals, 3, axis=0)
        else:
            raise Exception('No face normals')
        # color
        if self.geom.rgbs.ndim == 1:
            rgbs = np.tile(self.geom.rgbs, (self.count,1))
        else:
            if self.geom.faces is None:
                rgbs = self.geom.rgbs
            else:
                rgbs = self.geom.rgbs[self.geom.faces].reshape(-1,3)
        array = np.hstack([verts, normals, rgbs]).astype(np.float32)
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
        stride = 9 * 4 # float32 * 9
        # a_pos (0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        # a_normal (1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        # a_color (2)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(24))
        # unbind VAO
        gl.glBindVertexArray(0)