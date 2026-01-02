from one.scene.scene_object import SceneObject
from one.robot_sim.base.mech_structure import Link
from one.robot_sim.base.mech_state import MechState


class Scene:

    def __init__(self):
        self.dirty = True  # shader group needs update
        self._sobjs = []
        self._lnks = []
        self._states = []

    def __iter__(self): # for rendering order
        yield from self._sobjs
        yield from self._lnks

    def __getitem__(self, key): # for rendering order
        if key < len(self._sobjs):
            return self._sobjs[key]
        else:
            return self._lnks[key - len(self._sobjs)]

    def add(self, entity):
        if isinstance(entity, SceneObject):
            if entity not in self._sobjs:
                self._sobjs.append(entity)
                entity.scene = self
        elif isinstance(entity, MechState):
            self._states.append(entity)
            for lnk in entity.runtime_lnks:
                if lnk not in self._lnks:
                    self._lnks.append(lnk)
                    lnk.scene = self
        else:
            raise TypeError(f"Unsupported type: {type(entity)}")
        self.dirty = True

    def remove(self, entity):
        if isinstance(entity, SceneObject):
            if entity in self._sobjs:
                self._sobjs.remove(entity)
                entity.scene = None
        elif isinstance(entity, MechState):
            for lnk in entity.runtime_lnks:
                if lnk in self._lnks:
                    self._lnks.remove(lnk)
            if entity in self._states:
                self._states.remove(entity)
        self.dirty = True

    @property
    def sobjs(self):
        return tuple(self._sobjs)

    @property
    def lnks(self):
        return tuple(self._lnks)

    @property
    def states(self):
        return tuple(self._states)
