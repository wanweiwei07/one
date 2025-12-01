import pyglet.gl as gl
import wrs.viewer.device_buffer as db
import wrs.viewer.shader as sd


class Render:
    def __init__(self, camera):
        self.camera = camera
        self.shader = sd.Shader(sd.x_vert, sd.x_frag)
        self.shader.use()
        self._gl_setup()

    def set_shader(self, shader):
        self.shader = shader
        self.shader.use()

    def draw_model(self, model):
        self._set_uniforms(model)
        if model.geometry_buffer is None:
            model.geometry_buffer = db.DeviceBuffer(model.geometry)
        model.geometry_buffer.draw()

    def show(self, scene):
        for model in scene:
            self.draw_model(model)

    def _gl_setup(self):
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_MULTISAMPLE)

    def _set_uniforms(self, model):
        mvp_mat = self.camera.vp_mat @ model.wd_tfmat
        self.shader.program['u_mvp'] = mvp_mat.T.flatten()
        self.shader.program['u_model'] = model.wd_tfmat.T.flatten()
        self.program['u_point_size'] = 1.0
        self.shader.program['u_view_pos'] = self.camera.pos
        self.program['u_alpha'] = 1.0
