import pyglet.graphics as pg

x_vert = """
#version 330 core
// scaling is not allowed
// author: weiwei
// date: 20251127
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;
uniform mat4 u_mvp; //model-view-projection matrix
uniform mat4 u_model; //model matrix
uniform float u_point_size; //point size
out vec3 f_color;
out vec3 f_normal;
out vec3 v_pos;
void main() {
    f_color = a_color;
    f_normal = mat3(u_model) * a_normal;
    v_pos = vec3(u_model * vec4(a_pos, 1.0));
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    gl_PointSize = u_point_size;
}
"""

x_frag = """
#version 330 core
// ambient and point light only, 
// author: weiwei
// date: 20251127
in vec3 f_color;
in vec3 f_normal;
in vec3 v_pos;
out vec4 out_color;
uniform vec3 u_view_pos; //camera position in world space
uniform float u_alpha;
void main() {
    vec3 N = normalize(f_normal);
    // key / fill / rim light
    float dist = length(u_view_pos);
    vec3 L_key = normalize(u_view_pos + vec3(dist, 0.0, dist)-v_pos);
    vec3 L_fill = normalize(u_view_pos + vec3(-dist, 0.0, dist)-v_pos);
    vec3 L_rim = normalize(u_view_pos + vec3(0.0, 0.0, -dist)-v_pos);
    float diff = clamp(max(dot(N, L_key), 0.0) + 
                       max(dot(N, L_fill), 0.0) * 0.5 + 
                       max(dot(N, L_rim), 0.0) * 0.3, 0.0, 1.0);
    vec3 color = f_color * (vec3(0.1, 0.1, 0.1) + diff);
    out_color = vec4(color, u_alpha);
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
