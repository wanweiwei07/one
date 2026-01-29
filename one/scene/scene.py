from one.scene.scene_object import SceneObject
from one.robots.base.mech_base import MechBase


class Scene:

    def __init__(self):
        self.dirty = True  # shader group needs update
        self._sobjs = []
        self._lnks = []
        self._mecbas = []

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
                # entity.scene = self
        elif isinstance(entity, MechBase):
            if entity not in self._mecbas:
                self._mecbas.append(entity)
                for lnk in entity.runtime_lnks:
                    if lnk not in self._lnks:
                        self._lnks.append(lnk)
                        # lnk.scene = self
        else:
            raise TypeError(f"Unsupported type: {type(entity)}")
        self.dirty = True

    def remove(self, entity):
        if isinstance(entity, SceneObject):
            if entity in self._sobjs:
                self._sobjs.remove(entity)
                # entity.scene = None
        elif isinstance(entity, MechBase):
            for lnk in entity.runtime_lnks:
                if lnk in self._lnks:
                    self._lnks.remove(lnk)
            if entity in self._mecbas:
                self._mecbas.remove(entity)
        self.dirty = True


    @property
    def sobjs(self):
        return tuple(self._sobjs)

    @property
    def lnks(self):
        return tuple(self._lnks)

    @property
    def mecbas(self):
        return tuple(self._mecbas)
