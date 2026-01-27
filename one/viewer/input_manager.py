import pyglet.window.mouse as mouse
import pyglet.window.key as key


class InputManager:
    def __init__(self, window):
        self._window = window
        self.pressed_keys = set()
        self._pending_key_presses = set()
        self.pressed_buttons = set()
        window.push_handlers(self)

    def on_key_press(self, symbol, modifiers):
        self.pressed_keys.add(symbol)
        self._pending_key_presses.add(symbol)

    def on_key_release(self, symbol, modifiers):
        self.pressed_keys.discard(symbol)

    def on_mouse_press(self, x, y, button, modifiers):
        self.pressed_buttons.add(button)
        self.last_mouse_x = x
        self.last_mouse_y = y

    def on_mouse_release(self, x, y, button, modifiers):
        self.pressed_buttons.discard(button)
        self.last_mouse_x = None
        self.last_mouse_y = None

    def is_key_pressed(self, symbol):
        return symbol in self.pressed_keys

    def is_key_pressed_edge(self, symbol):
        if symbol in self._pending_key_presses:
            self._pending_key_presses.discard(symbol)
            return True
        return False

    def is_button_pressed(self, button):
        return button in self.pressed_buttons

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if mouse.LEFT in self.pressed_buttons:
            self._window.camera.mouse_orbit(dx, dy)
        elif mouse.MIDDLE in self.pressed_buttons:
            self._window.camera.mouse_pan(dx, dy)
        self.last_mouse_x = x
        self.last_mouse_y = y

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self._window.camera.mouse_zoom(scroll_y)
