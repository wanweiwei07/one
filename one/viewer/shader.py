import pyglet.graphics as pg

mesh_vert = """
#version 330 core
// vertex shader for mesh rendering
// author: weiwei
// date: 20251127
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 i_model0;
layout(location = 3) in vec4 i_model1;
layout(location = 4) in vec4 i_model2;
layout(location = 5) in vec4 i_model3;
layout(location = 6) in vec3 a_inst_rgba;
uniform mat4 u_view; //camera view matrix
uniform mat4 u_proj; //camera projection matrix
out vec3 v_normal;
out vec3 v_pos;
out vec3 v_rgba;
void main() {
    mat4 model = mat4(i_model0, i_model1, i_model2, i_model3);
    v_normal = mat3(model) * a_normal;
    v_pos = vec3(model * vec4(a_pos, 1.0));
    v_rgba = a_inst_rgba;
    gl_Position = u_proj * u_view * model * vec4(a_pos, 1.0);
}
"""

mesh_frag = """
#version 330 core
// ambient and point light only,
// author: weiwei
// date: 20251127
in vec3 v_normal;
in vec3 v_pos;
in vec4 v_rgba;
out vec4 out_color;
uniform vec3 u_view_pos; //camera position in world space
void main() {
    vec3 N = normalize(v_normal);
    // key / fill / rim light
    float dist = length(u_view_pos);
    vec3 L_key = normalize(u_view_pos + vec3(dist, 0.0, dist)-v_pos);
    vec3 L_fill = normalize(u_view_pos + vec3(-dist, 0.0, dist)-v_pos);
    vec3 L_rim = normalize(u_view_pos + vec3(0.0, 0.0, -dist)-v_pos);
    float diff = max(dot(N, L_key), 0.0) +
                 max(dot(N, L_fill), 0.0) * 0.5 +
                 max(dot(N, L_rim), 0.0) * 0.3;
    vec3 rgb = clamp(v_rgba.rgb * (vec3(0.2, 0.2, 0.2) + diff), 0.0, 1.0);
    out_color = vec4(rgb, v_rgba.a);
}
"""

pcd_vert = """
#version 330 core
// vertex shader for point cloud rendering
// author: weiwei
// date: 20251203
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_rgb;
uniform mat4 u_view; //camera view matrix
uniform mat4 u_proj; //camera projection matrix
uniform mat4 u_model; //model matrix
out vec3 v_rgb;
void main() {
    v_rgb = a_rgb;
    gl_Position = u_proj * u_view * u_model * vec4(a_pos, 1.0);
    gl_PointSize = 5;
}
"""

pcd_frag = """
#version 330 core
// point cloud shader for point cloud rendering
// author: weiwei
// date: 20251203
in vec3 v_rgb;
out vec4 out_color;
void main() {
    out_color = vec4(v_rgb, 1.0);
}
"""


class Shader:
    def __init__(self, vert_src, frag_src):
        self.vertex_shader = pg.shader.Shader(vert_src, "vertex")
        self.fragment_shader = pg.shader.Shader(frag_src, "fragment")
        self.program = pg.shader.ShaderProgram(self.vertex_shader, self.fragment_shader)

    def __setitem__(self, key, value):
        self.program[key] = value

    def use(self):
        self.program.use()
