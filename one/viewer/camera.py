import numpy as np
import pyglet.math as pm
import one.utils.decorators as decorators
import one.utils.math as rm
import one.scene.node as nd


class Camera(nd.Node):

    def __init__(self,
                 pos=(2, 2, 2),
                 look_at=(0, 0, 0),
                 up=(0, 0, 1),
                 fov=60,
                 aspect=1.7778,
                 near=0.01,
                 far=1000.0,
                 parent=None):
        self._pos = np.asarray(pos, dtype=np.float32)
        self._look_at = np.asarray(look_at, dtype=np.float32)
        self._up = np.asarray(up, dtype=np.float32)
        self._rotmat = rm.rotmat_from_look_at(pos=self._pos, target=self._look_at, up=self._up)
        super().__init__(rotmat=self._rotmat, pos=self._pos, parent=parent)
        self._fov = fov
        self._aspect = aspect  # default 16:9
        self._near = near
        self._far = far
        # cached matrices
        self._proj_mat = None
        self._proj_dirty = True

    def set_to(self, pos=None, look_at=None, up=None):
        if pos is not None:
            self._pos = np.asarray(pos, dtype=np.float32)
        if look_at is not None:
            self._look_at = np.asarray(look_at, dtype=np.float32)
        if up is not None:
            self._up = np.asarray(up, dtype=np.float32)
        if pos is not None or look_at is not None or up is not None:
            self.set_pose(rotmat=self._rotmat, pos=self._pos)

    def orbit(self, dt=None, axis=(0, 0, 1), angle_rad=rm.np.pi / 360):
        direction = self._pos - self._look_at
        rotate_around_rotmat = rm.rotmat_from_axangle(axis, angle_rad)
        direction_rotated = rotate_around_rotmat @ direction
        self._pos = self._look_at + direction_rotated
        rotmat = rm.rotmat_from_look_at(pos=self._pos, target=self._look_at, up=self._up)
        self.set_pose(rotmat=rotmat, pos=self._pos)

    def update(self):
        """Overwrite Node.update to compute world transforms considering the look_at functionality."""
        if not self._dirty:
            return
        self._rotmat = rm.rotmat_from_look_at(pos=self._pos, target=self._look_at, up=self._up)
        if self.parent is None:
            self._wd_rotmat = self._rotmat.copy()
            self._wd_pos = self._pos.copy()
        else:
            self.parent.update()
            self._wd_rotmat = self.parent._wd_rotmat @ self._rotmat
            self._wd_pos = self.parent._wd_rotmat @ self._pos + self.parent._wd_pos
        # update world 4x4
        self._wd_tfmat = rm.tfmat_from_rotmat_pos(self._wd_rotmat, self._wd_pos)
        self._dirty = False

    def update_proj(self, width=None, height=None):
        if width is not None and height is not None:
            self._aspect = width / height
        self._proj_mat = np.array(pm.Mat4.perspective_projection(aspect=self._aspect,
                                                                 z_near=self._near,
                                                                 z_far=self._far,
                                                                 fov=self._fov)).reshape(4, 4).T
        self._proj_dirty = False

    def _mark_proj_dirty(self):
        self._proj_dirty = True

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    @decorators.mark_dirty('_mark_dirty')
    def look_at(self, look_at):
        self._look_at = np.asarray(look_at, dtype=np.float32)

    @property
    def up(self):
        return self._up

    @up.setter
    @decorators.mark_dirty('_mark_dirty')
    def up(self, up):
        self._up = np.asarray(up, dtype=np.float32)

    @property
    def fov(self):
        return self._fov

    @fov.setter
    @decorators.mark_dirty('_proj_dirty')
    def fov(self, fov):
        self._fov = fov

    @property
    def near(self):
        return self._near

    @near.setter
    @decorators.mark_dirty('_proj_dirty')
    def near(self, near):
        self._near = near

    @property
    def far(self):
        return self._far

    @far.setter
    @decorators.mark_dirty('_proj_dirty')
    def far(self, far):
        self._far = far

    # getters for matrices, setting matrices should be done via other methods
    @property
    @decorators.lazy_update('_dirty', 'update_view')
    def view_mat(self):
        return self._rotmat.inverse()

    @property
    @decorators.lazy_update('_proj_dirty', 'update_proj')
    def proj_mat(self):
        return self._proj_mat

    @property
    def vp_mat(self):
        return self.proj_mat @ self.view_mat
