import numpy as np
import pyglet.gl as gl
import one.viewer.shader as sd
import one.viewer.screen_quad as sq


class Render:

    def __init__(self, camera):
        self.camera = camera
        self.mesh_shader = sd.Shader(sd.mesh_vert, sd.mesh_frag)
        self.pcd_shader = sd.Shader(sd.pcd_vert, sd.pcd_frag)
        self.tex_shader = sd.Shader(sd.tex_vert, sd.tex_frag)
        self.screen_quad = sq.ScreenQuad()
        self._groups_cache = None
        self._gl_setup()
        self._tmp = np.zeros(16, dtype=np.float32)  # for flattening matrices

    def draw(self, scene):
        cam_view_flatten = self.camera.view_mat.T.flatten()
        cam_proj_flatten = self.camera.proj_mat.T.flatten()
        if scene._dirty or self._groups_cache is None:
            self._groups_cache = self._build_shader_groups(scene)
            scene._dirty = False
        # mesh groups
        mesh_groups = self._groups_cache[self.mesh_shader]
        if mesh_groups:
            # mesh
            self.mesh_shader.use()
            self.mesh_shader.program["u_view"] = cam_view_flatten
            self.mesh_shader.program["u_proj"] = cam_proj_flatten
            self.mesh_shader.program["u_view_pos"] = self.camera.pos
            for instance_list in mesh_groups.values():
                tf_mat_array = np.empty((len(instance_list), 4, 4), np.float32)
                rgba_array = np.empty((len(instance_list), 4), np.float32)
                for i, (model, node) in enumerate(instance_list):
                    tf_mat_array[i] = (node.wd_tfmat @ model.tfmat).T
                    rgba_array[i] = np.array(
                        [*model.rgb, model.alpha], dtype=np.float32
                    )
                device_buffer = instance_list[0][0].get_device_buffer()
                device_buffer.update_instances(tf_mat_array, rgba_array)
                device_buffer.draw_instanced()
        # point cloud groups
        pcd_groups = self._groups_cache[self.pcd_shader]
        if pcd_groups:
            self.pcd_shader.use()
            self.pcd_shader.program["u_view"] = cam_view_flatten
            self.pcd_shader.program["u_proj"] = cam_proj_flatten
            for instance_list in pcd_groups.values():
                for model, node in instance_list:
                    self.pcd_shader.program["u_model"] = (
                            node.wd_tfmat @ model.local_tfmat
                    ).T.ravel()
                    model.get_device_buffer().draw()

    def draw_screen_quad(self, color_tex, width, height):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.tex_shader.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        self.tex_shader.program["u_color"] = 0
        self.tex_shader.program["u_texel"] = (1.0/width, 1.0/height)
        self.screen_quad.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _build_shader_groups(self, scene):
        groups = {self.mesh_shader: {}, self.pcd_shader: {}}
        for scn_obj in scene:
            for model in scn_obj.visuals:
                shader = self._pick_shader(model)
                device_buffer = model.get_device_buffer()
                if device_buffer.vao not in groups[shader]:
                    groups[shader][device_buffer.vao] = []
                groups[shader][device_buffer.vao].append((model, scn_obj.node))
        return groups

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

    def _pick_shader(self, model):
        if model.shader is not None:
            return model.shader
        if model.geometry.faces is not None:
            return self.mesh_shader
        else:
            return self.pcd_shader

