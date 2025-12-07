import pyglet.gl as gl
import one.viewer.shader as sd


class Render:
    def __init__(self, camera):
        self.camera = camera
        self.mesh_shader = sd.Shader(sd.mesh_vert, sd.mesh_frag)
        self.pcd_shader = sd.Shader(sd.pcd_vert, sd.pcd_frag)
        self._gl_setup()

    def draw_model(self, model, final_tfmat):
        device_buffer = model.get_device_buffer()
        self._set_uniforms(model, final_tfmat)
        device_buffer.draw()

    def show(self, scene):
        for entity in scene:
            for model in entity.visuals:
                final_tfmat = entity.node.wd_tfmat @ model.local_tfmat
                self.draw_model(model, final_tfmat)

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

    def _set_uniforms(self, model, model_final_tfmat):
        shader = self._pick_shader(model)
        shader.use()
        mvp_mat = self.camera.vp_mat @ model_final_tfmat
        if shader is self.mesh_shader:
            self.mesh_shader.use()
            self.mesh_shader.program['u_mvp'] = mvp_mat.T.flatten()
            self.mesh_shader.program['u_model'] = model_final_tfmat.T.flatten()
            self.mesh_shader.program['u_view_pos'] = self.camera.pos
            self.mesh_shader.program['u_rgb'] = model.rgb
            self.mesh_shader.program['u_alpha'] = 1.0
        elif shader is self.pcd_shader:
            self.pcd_shader.use()
            self.pcd_shader.program['u_mvp'] = mvp_mat.T.flatten()
            self.pcd_shader.program['u_alpha'] = 1.0
        else:
            raise ValueError("Unknown shader type!")

    def _pick_shader(self, model):
        if model.shader is not None:
            return model.shader
        geom = model.geometry
        if geom.faces is not None:
            return self.mesh_shader
        else:
            return self.pcd_shader
