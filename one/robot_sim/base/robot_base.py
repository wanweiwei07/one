import numpy as np
import one.utils.decorator as deco
import one.robot_sim.base.kinematic_state as rstate
import one.robot_sim.base.robot_structure as rstruct


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


class RobotBase:
    _structure: rstruct.RobotStructure = None

    @classmethod
    def _build_structure(cls):
        raise NotImplementedError

    def __init__(self, base_tfmat=None):
        self.kin_state = rstate.KinematicState(self.structure, root_tfmat=base_tfmat)
        self.home_qs = np.zeros(self.structure.flat.n_joints, dtype=np.float32)
        self._mountings: dict[object, Mounting] = {}
        self.fk(qs=self.home_qs, update=True)

    def fk(self, qs=None, update=True):
        """
        TODO: update visual and update collision
        forward kinematics, update cannot be true when root_tfmat is given
        :param qs:
        :param update: whether to update the mountings after fk
        :return:
        """
        wd_lnk_tfmat_arr = self.kin_state.fk(qs)
        if update:
            self.update()
        return wd_lnk_tfmat_arr

    def update(self):
        self.kin_state.update()
        for m in self._mountings.values():
            self._update_mounting(m)

    def attach_to(self, scene):
        return self.kin_state.attach_to(scene)

    def remove_from(self, scene):
        return self.kin_state.remove_from(scene)

    def get_link_wd_tfmat(self, link):
        return self.kin_state.get_link_reference_wd_tfmat(link)

    def mount(self, child, parent_link, engage_tfmat):
        if child in self._mountings or child is self:
            raise ValueError("Child already mounted or self-mounting")
        who = child
        if isinstance(child, RobotBase):
            who = child.kin_state.runtime_links[0]
        if who.scene is not self.kin_state.runtime_links[0].scene:
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
        new_obj.kin_state = self.kin_state.clone()
        new_obj._mountings = dict(self._mountings)
        return new_obj

    def _update_mounting(self, mounting: Mounting):
        parent_tfmat = self.get_link_wd_tfmat(mounting.parent_link)
        child_tfmat = parent_tfmat @ mounting.engage_tfmat
        if isinstance(mounting.child, RobotBase):
            mounting.child.kin_state.root_tfmat = child_tfmat
            mounting.child.fk(update=True)
        else:
            mounting.child.tfmat = child_tfmat

    @property
    @deco.readonly_view
    def qs(self):
        return self.kin_state.qs

    @property
    def structure(self):
        cls = type(self)
        if cls._structure is None:
            cls._structure = cls._build_structure()
        return cls._structure

    @property
    def toggle_render_collision(self):
        return self.kin_state.toggle_render_collision

    @toggle_render_collision.setter
    def toggle_render_collision(self, flag=True):
        self.kin_state.toggle_render_collision = flag

    @property
    @deco.readonly_view
    def base_tfmat(self):
        return self.kin_state.root_tfmat

    @base_tfmat.setter
    def base_tfmat(self, tfmat):
        self.kin_state.root_tfmat = tfmat
        self.kin_state.fk()

    @property
    @deco.readonly_view
    def base_pos(self):
        return self.kin_state.root_tfmat[:3, 3]

    @base_pos.setter
    def base_pos(self, pos):
        self.kin_state.root_tfmat[:3, 3] = pos
        self.kin_state.fk()

    @property
    @deco.readonly_view
    def base_rotmat(self):
        return self.kin_state.root_tfmat[:3, :3]

    @base_rotmat.setter
    def base_rotmat(self, rotmat):
        self.kin_state.root_tfmat[:3, :3] = rotmat
        self.kin_state.fk()
