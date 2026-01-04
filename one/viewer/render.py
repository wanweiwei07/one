import numpy as np
import pyglet.gl as gl
import one.viewer.shader as ovs
import one.viewer.screen_quad as ovsq


class Render:

    def __init__(self, camera):
        self.camera = camera
        self.mesh_shader = ovs.Shader(ovs.mesh_vert, ovs.mesh_matte_frag)
        self.pcd_shader = ovs.Shader(ovs.pcd_vert, ovs.pcd_frag)
        self.outline_shader = ovs.Shader(ovs.outline_vert, ovs.outline_frag)
        self.tex_shader = ovs.Shader(ovs.tex_vert, ovs.tex_frag)
        self.screen_quad = ovsq.ScreenQuad()
        self._groups_cache = None
        self._gl_setup()
        self._tmp = np.zeros(16, dtype=np.float32)  # for flattening matrices

    def draw(self, scene):
        cam_view = self.camera.view_mat.T.flatten()
        cam_proj = self.camera.projmat.T.flatten()
        # rebuild cache if needed
        if scene.dirty or self._groups_cache is None:
            self._groups_cache = self._build_shader_groups(scene)
            scene.dirty = False
        solid_group = self._groups_cache["mesh_solid"]
        transparent_group = self._groups_cache["mesh_transparent"]
        pcd_group = self._groups_cache["pcd"]
        self._draw_solid_mesh_with_outline(solid_group, cam_view, cam_proj)
        self._draw_transparent_mesh(transparent_group, cam_view, cam_proj)
        self._draw_pcd(pcd_group, cam_view, cam_proj)

    def draw_screen_quad(self, color_tex, width, height):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.tex_shader.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        self.tex_shader.program["u_color"] = 0
        self.tex_shader.program["u_texel"] = (1.0 / width, 1.0 / height)
        self.screen_quad.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _build_shader_groups(self, scene):
        groups = {"mesh_solid": {},  # vao -> [(model, node), ...]
                  "mesh_transparent": {},  # vao -> [(model, node), ...]
                  "pcd": {}}  # vao -> [(model, node), ...]
        for scn_obj in scene:
            for model in scn_obj.visuals:
                shader = self._pick_shader(model)
                device_buffer = model.geometry.get_device_buffer()
                if shader is self.mesh_shader:
                    target = (groups["mesh_transparent"]
                              if model.alpha < 0.999
                              else groups["mesh_solid"])
                else:
                    target = groups["pcd"]
                if device_buffer.vao not in target:
                    target[device_buffer.vao] = []
                target[device_buffer.vao].append((model, scn_obj.node))
            if scn_obj.toggle_render_collision:
                for c in scn_obj.collisions:
                    model = c.to_render_model()
                    shader = self._pick_shader(model)
                    device_buffer = model.geometry.get_device_buffer()
                    if shader is self.mesh_shader:
                        target = (groups["mesh_transparent"]
                                  if model.alpha < 0.999
                                  else groups["mesh_solid"])
                    else:
                        target = groups["pcd"]
                    if device_buffer.vao not in target:
                        target[device_buffer.vao] = []
                    target[device_buffer.vao].append((model, scn_obj.node))
        return groups

    def _draw_solid_mesh_with_outline(self, solid_groups, cam_view, cam_proj):
        if not solid_groups:
            return
        # normal pass
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
        gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
        gl.glStencilMask(0xFF)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glCullFace(gl.GL_BACK)
        self.mesh_shader.use()
        self.mesh_shader.program["u_view"] = cam_view
        self.mesh_shader.program["u_proj"] = cam_proj
        self.mesh_shader.program["u_view_pos"] = self.camera.pos
        for instance_list in solid_groups.values():
            tf = np.empty((len(instance_list), 4, 4), np.float32)
            rgba = np.empty((len(instance_list), 4), np.float32)
            for i, (model, node) in enumerate(instance_list):
                tf[i] = (node.wd_tfmat @ model.tfmat).T
                rgba[i] = (*model.rgb, model.alpha)
            device = instance_list[0][0].geometry.get_device_buffer()
            device.update_instances(tf, rgba)
            device.draw_instanced()
        # outline pass
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilFunc(gl.GL_NOTEQUAL, 1, 0xFF)
        gl.glStencilMask(0x00)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glCullFace(gl.GL_FRONT)
        self.outline_shader.use()
        self.outline_shader.program["u_view"] = cam_view
        self.outline_shader.program["u_proj"] = cam_proj
        for instance_list in solid_groups.values():
            tf = np.empty((len(instance_list), 4, 4), np.float32)
            for i, (model, node) in enumerate(instance_list):
                tf[i] = (node.wd_tfmat @ model.tfmat).T
            device = instance_list[0][0].geometry.get_device_buffer()
            device.update_instances(tf)
            device.draw_instanced()
        # restore state
        gl.glStencilMask(0xFF)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_STENCIL_TEST)
        gl.glCullFace(gl.GL_BACK)

    def _draw_transparent_mesh(self, transparent_groups, cam_view, cam_proj):
        if not transparent_groups:
            return
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_STENCIL_TEST)
        gl.glCullFace(gl.GL_BACK)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.mesh_shader.use()
        self.mesh_shader.program["u_view"] = cam_view
        self.mesh_shader.program["u_proj"] = cam_proj
        self.mesh_shader.program["u_view_pos"] = self.camera.pos
        cam_pos = np.asarray(self.camera.pos, dtype=np.float32)
        instances = []
        for instance_list in transparent_groups.values():
            for model, node in instance_list:
                world = node.wd_tfmat @ model.tfmat
                pos = world[:3, 3]
                d2 = float(np.dot(pos - cam_pos, pos - cam_pos))
                device = model.geometry.get_device_buffer()
                instances.append((d2, model, node, device))
        instances.sort(key=lambda x: x[0], reverse=True)
        for _, model, node, device in instances:
            tfmat = np.empty((1, 4, 4), np.float32)
            rgba = np.empty((1, 4), np.float32)
            tfmat[0] = (node.wd_tfmat @ model.tfmat).T
            rgba[0] = (*model.rgb, model.alpha)
            device.update_instances(tfmat, rgba)
            device.draw_instanced()
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_pcd(self, pcd_groups, cam_view, cam_proj):
        if not pcd_groups:
            return
        self.pcd_shader.use()
        self.pcd_shader.program["u_view"] = cam_view
        self.pcd_shader.program["u_proj"] = cam_proj
        for instance_list in pcd_groups.values():
            for model, node in instance_list:
                self.pcd_shader.program["u_model"] = (
                        node.wd_tfmat @ model.local_tfmat
                ).T.ravel()
                model.get_device_buffer().draw()

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
