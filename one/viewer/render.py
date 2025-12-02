import pyglet.gl as gl
import one.viewer.device_buffer as db
import one.viewer.shader as sd
import one.scene.geometry as geom
import one.scene.model as mdl
import one.utils.math as rm


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
        if isinstance(model, mdl.Model):
            self._set_uniforms(model, None)
            if model.geometry.device_buffer is None:
                model.geometry.device_buffer = db.DeviceBuffer(model.geometry)
            model.geometry.device_buffer.draw()
        elif isinstance(model, geom.GeometryBase):
            self._set_uniforms(model, rm.np.eye(4))
            if model.device_buffer is None:
                model.device_buffer = db.DeviceBuffer(model)
            model.device_buffer.draw()

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

    def _set_uniforms(self, model, model_tfmat=None):
        if model_tfmat is None:
            model_tfmat = model.wd_tfmat
        mvp_mat = self.camera.vp_mat @ model_tfmat
        self.shader.program['u_mvp'] = mvp_mat.T.flatten()
        self.shader.program['u_model'] = model_tfmat.T.flatten()
        self.shader.program['u_point_size'] = 1.0
        self.shader.program['u_view_pos'] = self.camera.pos
        self.shader.program['u_alpha'] = 1.0
