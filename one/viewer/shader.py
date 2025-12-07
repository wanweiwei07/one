import pyglet.graphics as pg

mesh_vert = """
#version 330 core
// vertex shader for mesh rendering
// author: weiwei
// date: 20251127
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_mvp; //model-view-projection matrix
uniform mat4 u_model; //model matrix
out vec3 v_normal;
out vec3 v_pos;
void main() {
    v_normal = mat3(u_model) * a_normal;
    v_pos = vec3(u_model * vec4(a_pos, 1.0));
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

mesh_frag = """
#version 330 core
// ambient and point light only,
// author: weiwei
// date: 20251127
in vec3 v_normal;
in vec3 v_pos;
out vec4 out_color;
uniform vec3 u_view_pos; //camera position in world space
uniform vec3 u_rgb;
uniform float u_alpha;
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
    vec3 rgb = clamp(u_rgb * (vec3(0.2, 0.2, 0.2) + diff), 0.0, 1.0);
    out_color = vec4(rgb, u_alpha);
}
"""

pcd_vert = """
#version 330 core
// vertex shader for point cloud rendering
// author: weiwei
// date: 20251203
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_rgb;
uniform mat4 u_mvp; //model-view-projection matrix
out vec3 v_rgb;
void main() {
    v_rgb = a_rgb;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
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
uniform float u_alpha;
void main() {
    out_color = vec4(v_rgb, u_alpha);
}
"""


class Shader:
    def __init__(self, vert_src, frag_src):
        self.vertex_shader = pg.shader.Shader(vert_src, 'vertex')
        self.fragment_shader = pg.shader.Shader(frag_src, 'fragment')
        self.program = pg.shader.ShaderProgram(self.vertex_shader, self.fragment_shader)

    def __setitem__(self, key, value):
        self.program[key] = value

    def use(self):
        self.program.use()
