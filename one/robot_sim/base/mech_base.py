import numpy as np
import one.utils.decorator as deco
import one.robot_sim.base.mech_state as rstate
import one.robot_sim.base.mech_structure as rstruct


# TODO: dataclass, frozen, slots
class Mounting:
    def __init__(self, child, parent_link, engage_tfmat):
        self.child = child
        self.parent_link = parent_link
        self.engage_tfmat = engage_tfmat


# TODO : implement Anchor if needed
# class Anchor:
#     def __init__(self, parent_link, local_tfmat=np.eye(4, dtype=np.float32)):
#         self.parent_link = parent_link
#         self.local_tfmat = local_tfmat
#
#     @property
#     @deco.readonly_view
#     def tfmat(self):
#         return self.parent_link.get_reference_wd_tfmat() @ self.local_tfmat


class MechBase:
    _structure: rstruct.MechStruct = None

    @classmethod
    def _build_structure(cls):
        raise NotImplementedError

    def __init__(self, base_rotmat=None, base_pos=None):
        self.state = rstate.MechState(self.structure, base_rotmat=base_rotmat, base_pos=base_pos)
        self.home_qs = np.zeros(self.state.n_jnts, dtype=np.float32)
        self._mountings: dict[object, Mounting] = {}
        self.fk(qs=self.home_qs, update=True)

    def set_base_rotmat_pos(self, rotmat, pos):
        self.state.base_rotmat[:] = rotmat
        self.state.base_pos[:] = pos
        self.fk(update=True)

    def fk(self, qs=None, update=True):
        """
        TODO: update visual and update collision
        forward kinematics, update cannot be true when root_tfmat is given
        :param qs:
        :param update: whether to update the mountings after fk
        :return:
        """
        wd_lnk_tfmat_arr = self.state.fk(qs)
        if update:
            self.update()
        return wd_lnk_tfmat_arr

    def update(self):
        self.state.update()
        for m in self._mountings.values():
            self._update_mounting(m)

    def attach_to(self, scene):
        self.state.attach_to(scene)

    def remove_from(self, scene):
        return self.state.remove_from(scene)

    def get_link_wd_tfmat(self, link):
        return self.state.get_lnk_ref_tfmat(link)

    def mount(self, child, parent_link, engage_tfmat):
        if child in self._mountings or child is self:
            raise ValueError("Child already mounted or self-mounting")
        who = child
        if isinstance(child, MechBase):
            who = child.state.runtime_lnks[0]
        if who.scene is not self.state.runtime_lnks[0].scene:
            raise ValueError("Child object not in the same scene")
        if engage_tfmat is None:
            engage_tfmat = np.eye(4, dtype=np.float32)
        else:
            engage_tfmat = np.asarray(engage_tfmat, dtype=np.float32)
        self._mountings[child] = Mounting(child, parent_link, engage_tfmat)

    def unmount(self, child):
        try:
            return self._mountings.pop(child)
        except KeyError:
            raise ValueError("Child not mounted")

    def clone(self):
        new_obj = self.__class__.__new__(self.__class__)
        new_obj.home_qs = self.home_qs.copy()
        new_obj.state = self.state.clone()
        new_obj._mountings = dict(self._mountings)
        return new_obj

    @property
    @deco.readonly_view
    def qs(self):
        return self.state.qs

    @property
    def structure(self):
        cls = type(self)
        if cls._structure is None:
            cls._structure = cls._build_structure()
        return cls._structure

    @property
    def toggle_render_collision(self):
        return self.state.toggle_render_collision

    @toggle_render_collision.setter
    def toggle_render_collision(self, flag=True):
        self.state.toggle_render_collision = flag

    @property
    @deco.readonly_view
    def base_tfmat(self):
        return self.state.base_tfmat

    @base_tfmat.setter
    def base_tfmat(self, tfmat):
        self.state.base_tfmat = tfmat
        self.state.fk()

    @property
    @deco.readonly_view
    def base_pos(self):
        return self.state.base_tfmat[:3, 3]

    @base_pos.setter
    def base_pos(self, pos):
        self.state.base_tfmat[:3, 3] = pos
        self.fk(update=True)

    @property
    @deco.readonly_view
    def base_rotmat(self):
        return self.state.base_tfmat[:3, :3]

    @base_rotmat.setter
    def base_rotmat(self, rotmat):
        self.state.base_tfmat[:3, :3] = rotmat
        self.state.fk()

    @property
    def rgba(self):
        return self.state.rgba

    @rgba.setter
    def rgba(self, value):
        self.state.rgba = value

    @property
    def rgb(self):
        return self.state.rgb

    @rgb.setter
    def rgb(self, value):
        self.state.rgb = value

    @property
    def alpha(self):
        return self.state.alpha

    @alpha.setter
    def alpha(self, value):
        self.state.alpha = value

    def _update_mounting(self, mounting: Mounting):
        parent_tfmat = self.get_link_wd_tfmat(mounting.parent_link)
        child_tfmat = parent_tfmat @ mounting.engage_tfmat
        if isinstance(mounting.child, MechBase):
            mounting.child.state.base_tfmat = child_tfmat
            mounting.child.fk(update=True)
        else:
            mounting.child.tfmat = child_tfmat
