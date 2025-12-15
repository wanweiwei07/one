import pyglet.gl as gl


class RenderTarget:
    def __init__(self, width, height, samples=4):
        self.width = width
        self.height = height
        self.samples=samples
        # final
        self.fbo = 0
        self.color_tex = 0
        self.depth_rb = 0
        # MSAA
        self.msaa_fbo = 0
        self.msaa_color_rb = 0
        self.msaa_depth_rb = 0
        self._build()

    def bind(self):
        if self.samples > 1:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.msaa_fbo)
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def unbind(self):
        if self.samples > 1:
            self._resolve_msaa()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _build(self):
        # create fbo
        fbo = (gl.GLuint * 1)()
        gl.glGenFramebuffers(1, fbo)
        self.fbo = fbo[0]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        # create color texture
        color_tex = (gl.GLuint * 1)()
        gl.glGenTextures(1, color_tex)
        self.color_tex = color_tex[0]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
            self.width, self.height, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.color_tex,
            0
        )
        # create depth renderbuffer
        depth_rb = (gl.GLuint * 1)()
        gl.glGenRenderbuffers(1, depth_rb)
        self.depth_rb = depth_rb[0]
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.depth_rb)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER,
            gl.GL_DEPTH_COMPONENT24,
            self.width, self.height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self.depth_rb
        )
        assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if self.samples > 1:
            # MSAA framebuffer
            fbo = (gl.GLuint * 1)()
            gl.glGenFramebuffers(1, fbo)
            self.msaa_fbo = fbo[0]
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.msaa_fbo)
            # MSAA color renderbuffer
            rb = (gl.GLuint * 1)()
            gl.glGenRenderbuffers(1, rb)
            self.msaa_color_rb = rb[0]
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.msaa_color_rb)
            gl.glRenderbufferStorageMultisample(
                gl.GL_RENDERBUFFER,
                self.samples,
                gl.GL_RGBA8,
                self.width,
                self.height
            )
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_RENDERBUFFER,
                self.msaa_color_rb
            )
            # MSAA depth renderbuffer
            rb = (gl.GLuint * 1)()
            gl.glGenRenderbuffers(1, rb)
            self.msaa_depth_rb = rb[0]
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.msaa_depth_rb)
            gl.glRenderbufferStorageMultisample(
                gl.GL_RENDERBUFFER,
                self.samples,
                gl.GL_DEPTH_COMPONENT24,
                self.width,
                self.height
            )
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER,
                gl.GL_DEPTH_ATTACHMENT,
                gl.GL_RENDERBUFFER,
                self.msaa_depth_rb
            )
            assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _resolve_msaa(self):
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.msaa_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo)
        gl.glBlitFramebuffer(
            0, 0, self.width, self.height,
            0, 0, self.width, self.height,
            gl.GL_COLOR_BUFFER_BIT,
            gl.GL_NEAREST
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)