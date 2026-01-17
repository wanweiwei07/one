from pyglet.gl import gl_info
print("OpenGL version:", gl_info.get_version())
print("GLSL version:", gl_info.get_version_string())