import pyglet
import pyglet.gl as gl
import one.viewer.render as rd
import one.viewer.camera as cam
import one.viewer.input_manager as input_mgr

config = pyglet.gl.Config(
    major_version=4,
    minor_version=6,
    double_buffer=True,
    depth_size=24,
    sample_buffers=1,  # multisample
    samples=4,  # MSAA 4X
)


class World(pyglet.window.Window):
    def __init__(self,
                 cam_pos=(.1, .1, .1),
                 cam_lookat_pos=(0, 0, 0),
                 width=1920,
                 height=1080,
                 toggle_auto_cam_orbit=False):
        super().__init__(width, height, config=config, resizable=True)
        self.set_caption("WRS World")
        self.set_location(100, 100)
        self.camera = cam.Camera(pos=cam_pos, look_at=cam_lookat_pos, aspect=width / height)
        self.render = rd.Render(camera=self.camera)
        # self.fps_display = pyglet.window.FPSDisplay(self)
        self.scene = None
        if toggle_auto_cam_orbit:
            self.schedule_interval(self.auto_cam_orbit, interval=1 / 30.0)
        self.input_manager = input_mgr.InputManager(self)

    def on_resize(self, width, height):
        gl.glViewport(0, 0, *self.get_framebuffer_size())
        self.camera.update_proj(width, height)

    def on_draw(self):
        self.clear()
        if self.scene is not None:
            self.render.show(self.scene)
        # self.fps_display.draw()

    def set_scene(self, scene):
        self.scene = scene

    def auto_cam_orbit(self, dt, interval=1 / 30.0):
        self.camera.orbit(angle_rad=interval)

    def schedule_interval(self, function, interval=1 / 30.0):
        pyglet.clock.schedule_interval(function, interval=interval)

    def run(self):
        pyglet.app.run()


if __name__ == '__main__':
    import trimesh as trm
    from one import mdl, geom, scn, const

    mesh = trm.load_mesh("bunnysim.stl")
    model = mdl.Model(geom.GeometryBase(verts=mesh.vertices,
                                        faces=mesh.faces,
                                        rgbs=const.BasicColor.GREEN))
    scene = scn.Scene()
    scene.add(model)
    base = World()
    base.set_scene(scene)
    base.run()