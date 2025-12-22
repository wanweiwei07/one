import numpy as np
import one.utils.decorator as deco
import one.robot_sim.base.kinematic_state as rstate
import one.robot_sim.base.robot_structure as rstruct


class Mounting:
    def __init__(self, child, parent_link, engage_tfmat):
        self.child = child
        self.parent_link = parent_link
        self.engage_tfmat = engage_tfmat


class RobotBase:
    _structure: rstruct.RobotStructure = None

    @classmethod
    def _build_structure(cls):
        raise NotImplementedError

    def __init__(self):
        self.kin_state = rstate.KinematicState(self.structure)
        self.home_qs = np.zeros(self.structure.flat.n_joints, dtype=np.float32)
        self._mountings: dict[object, Mounting] = {}
        self.fk(qs=self.home_qs, update=True)

    def fk(self, qs=None, root_tfmat=None, update=True):
        """
        forward kinematics, update cannot be true when root_tfmat is given
        :param qs:
        :param root_tfmat: Allow specifying root (4,4). base_tfmat will be ignored if given.
        :param update: whether to update the mountings after fk
        :return:
        """
        wd_lnk_tfmat_arr = self.kin_state.fk(qs, root_tfmat)
        if update:
            if root_tfmat is not None:
                raise ValueError("Cannot update mountings when root_tfmat is given")
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
            mounting.child.kin_state.base_tfmat = child_tfmat
            mounting.child.fk(update=True)
        else:
            mounting.child.set_tfmat(child_tfmat)

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
