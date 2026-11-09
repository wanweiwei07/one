"""Dear ImGui panels drawn over a :class:`one.viewer.world.World`.

Uses imgui-bundle (``pip install imgui-bundle``) and its pyglet backend,
imported only by this module, so the rest of the viewer does not need it.
``draw`` is called once per frame between ImGui's new_frame and render, to
issue the ``imgui.*`` calls of the panels::

    from imgui_bundle import imgui
    import one.viewer.imgui_overlay as ovio

    def panel():
        imgui.begin('tools')
        if imgui.button('go'):
            ...
        imgui.end()

    ovio.ImGuiOverlay(base, panel)

A mouse press / drag / scroll over a panel goes to the panel only, the
camera stays put; off the panels the camera works as before. Keys still
reach the World's InputManager.
"""

import pyglet
from imgui_bundle import imgui
from imgui_bundle.python_backends.pyglet_backend import create_renderer


class _GuardMouse:
    """Sits between the World's InputManager (below) and ImGui (above): a
    mouse event over a panel stops here, so the camera does not move."""

    def _over_panel(self, *args):
        if imgui.get_io().want_capture_mouse:
            return pyglet.event.EVENT_HANDLED
        return None

    on_mouse_press = on_mouse_release = _over_panel
    on_mouse_drag = on_mouse_scroll = _over_panel


class ImGuiOverlay:
    """ImGui over ``world``: ``draw()`` each frame issues the panels.
    ``scale``: size of the text and widgets, default the screen's DPI scale
    (1.5 on a 150 % display), so they keep their usual size on screen."""

    def __init__(self, world, draw, scale=None):
        self.world, self.draw = world, draw
        imgui.create_context()
        imgui.get_io().set_ini_filename('')       # no imgui.ini dropped in cwd
        scale = float(getattr(world, 'scale', 1.0) if scale is None
                      else scale)
        style = imgui.get_style()
        style.scale_all_sizes(scale)
        style.font_scale_dpi = scale
        world.push_handlers(_GuardMouse())        # above the InputManager
        self.renderer = create_renderer(world)    # above the guard
        world.overlays.append(self.frame)

    def frame(self):
        io = imgui.get_io()
        # the backend sets the size only on a resize event (the window's
        # first one comes before it is attached), and takes the screen's DPI
        # scale for the framebuffer's, though on Windows the window size is
        # already in pixels -- the panels then land off-screen. Both from
        # the real sizes, every frame.
        (fw, fh), (ww, wh) = (self.world.get_framebuffer_size(),
                              self.world.get_size())
        io.display_size = (ww, wh)
        io.display_framebuffer_scale = (fw / ww, fh / wh)
        self.renderer.process_inputs()
        imgui.new_frame()
        self.draw()
        imgui.render()
        self.renderer.render(imgui.get_draw_data())
