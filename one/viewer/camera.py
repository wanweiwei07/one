import numpy as np
import pyglet.math as pm
import one.utils.decorator as deco
import one.utils.math as rm
import one.scene.scene_node as nd


class Camera(nd.SceneNode):

    def __init__(self,
                 pos=(2, 2, 2),
                 look_at=(0, 0, 0),
                 up=(0, 0, 1),
                 fov=45,
                 aspect=1.7778,
                 near=0.01,
                 far=1000.0,
                 parent=None):
        self._pos = np.asarray(pos, dtype=np.float32)
        self._look_at = np.asarray(look_at, dtype=np.float32)
        self._up = np.asarray(self._fix_up_vector(self._pos, self._look_at, up),
                              dtype=np.float32)
        self._rotmat = rm.rotmat_from_look_at(pos=self._pos,
                                              look_at=self._look_at,
                                              up=self._up)
        super().__init__(rotmat=self._rotmat, pos=self._pos, parent=parent)
        self._fov = fov
        self._aspect = aspect  # default 16:9
        self._near = near
        self._far = far
        # cached
        self._projmat = None
        self._proj_dirty = True

    def set_to(self, pos, look_at, up=None):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.look_at = np.asarray(look_at, dtype=np.float32)
        if up is not None:
            self.up = np.asarray(up, dtype=np.float32)

    def orbit(self, axis=(0, 0, 1), angle_rad=np.pi / 360):
        direction = self._pos - self._look_at
        rotmat = rm.rotmat_from_axangle(axis, angle_rad)
        direction_rotated = rotmat @ direction
        self._pos = self._look_at + direction_rotated
        self._up = (rotmat @ self._up)
        self._up /= np.linalg.norm(self._up)
        self._up = self._fix_up_vector(self._pos, self._look_at, self._up)
        self._dirty = True

    def mouse_orbit(self, dx, dy, sensitivity=0.002):
        right_axis = self.wd_tfmat[:3, 0]
        up_axis = self.wd_tfmat[:3, 1]
        self.orbit(axis=up_axis, angle_rad=-dx * sensitivity)
        self.orbit(axis=right_axis, angle_rad=dy * sensitivity)

    def mouse_pan(self, dx, dy, sensitivity=0.0003):
        right_axis = self.wd_tfmat[:3, 0]
        up_axis = self.wd_tfmat[:3, 1]
        self.pos = self.pos - right_axis * dx * sensitivity - up_axis * dy * sensitivity
        self.look_at = self.look_at - right_axis * dx * sensitivity - up_axis * dy * sensitivity

    def mouse_zoom(self, delta, sensitivity=0.05):
        direction = self.pos - self.look_at
        zoom_amount = delta * sensitivity
        self.pos = self.pos + direction * zoom_amount

    @nd.SceneNode.rotmat.setter
    def rotmat(self, rotmat):
        """Disable direct setting of rotmat on Camera."""
        raise AttributeError("Cannot set rotmat directly on Camera. Use set_to() method instead.")

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    @deco.mark_dirty('_mark_dirty')
    def look_at(self, look_at):
        self._look_at = np.asarray(look_at, dtype=np.float32)

    @property
    def up(self):
        return self._up

    @up.setter
    @deco.mark_dirty('_mark_dirty')
    def up(self, up):
        self._up = np.asarray(up, dtype=np.float32)

    @property
    def fov(self):
        return self._fov

    @fov.setter
    @deco.mark_dirty('_proj_dirty')
    def fov(self, fov):
        self._fov = fov

    @property
    def near(self):
        return self._near

    @near.setter
    @deco.mark_dirty('_proj_dirty')
    def near(self, near):
        self._near = near

    @property
    def far(self):
        return self._far

    @far.setter
    @deco.mark_dirty('_proj_dirty')
    def far(self, far):
        self._far = far

    # getters for matrices, setting matrices should be done via other methods
    @property
    @deco.lazy_update('_dirty', '_rebuild_tfmat')
    def view_mat(self):
        return rm.tfmat_inverse(self._wd_tfmat)

    @property
    @deco.lazy_update('_proj_dirty', '_rebuild_projmat')
    def projmat(self):
        return self._projmat

    def _mark_proj_dirty(self):
        self._proj_dirty = True

    def _fix_up_vector(self, pos, look_at, up):
        # TODO: elevate to utils.math
        fwd_length, fwd = rm.unit_vec(look_at - pos)
        up_length, up = rm.unit_vec(up)
        dot_val = np.dot(fwd, up)
        limit = 0.99 * (fwd_length * up_length)
        if dot_val > limit:
            if np.allclose(up, (0, 0, 1)):
                up = (1, 0, 0)
            else:
                up = (0, 0, 1)
        return up

    def _rebuild_tfmat(self):
        if not self._dirty:
            return
        self._up = np.asarray(self._fix_up_vector(self._pos, self._look_at, self._up),
                              dtype=np.float32)
        self._rotmat = rm.rotmat_from_look_at(pos=self._pos,
                                              look_at=self._look_at,
                                              up=self._up)
        super()._rebuild_tfmat()

    def _rebuild_projmat(self, width=None, height=None):
        if width is not None and height is not None:
            self._aspect = width / height
        self._projmat = np.array(pm.Mat4.perspective_projection(aspect=self._aspect,
                                                                z_near=self._near,
                                                                z_far=self._far,
                                                                fov=self._fov)).reshape(4, 4).T
        self._proj_dirty = False
