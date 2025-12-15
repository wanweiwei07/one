import numpy as np
import pyglet.gl as gl
import ctypes


class ScreenQuad:
    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self._build()

    def _build(self):
        verts = np.array([
            -1, -1,
            1, -1,
            1, 1,
            -1, -1,
            1, 1,
            -1, 1,
        ], dtype=np.float32)
        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self.vao = vao[0]
        vbo = (gl.GLuint * 1)()
        gl.glGenBuffers(1, vbo)
        self.vbo = vbo[0]
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            verts.nbytes,
            verts.ctypes.data,
            gl.GL_STATIC_DRAW
        )
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 2, gl.GL_FLOAT, False, 2 * 4, ctypes.c_void_p(0)
        )
        gl.glBindVertexArray(0)

    def draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
