from one.scene.scene_object import SceneObject
from one.robot_sim.base.robot_structure import Link
from one.robot_sim.base.robot_structure import RobotStructure


class Scene:

    def __init__(self):
        self.dirty = True  # shader group needs update
        self._scene_objects = []
        self._links = []
        self._robot_structures = []

    def __iter__(self):
        yield from self._scene_objects
        yield from self._links

    def __getitem__(self, key):
        if key < len(self._scene_objects):
            return self._scene_objects[key]
        else:
            return self._links[key - len(self._scene_objects)]

    def add(self, entity):
        if isinstance(entity, Link):
            if entity not in self._links:
                self._links.append(entity)
                entity.scene = self
        elif isinstance(entity, SceneObject):
            if entity not in self._scene_objects:
                self._scene_objects.append(entity)
                entity.scene = self
        elif isinstance(entity, RobotStructure):
            self._robot_structures.append(entity)
        else:
            raise TypeError(f"Unsupported type: {type(entity)}")
        self.dirty = True

    def remove(self, entity):
        if isinstance(entity, SceneObject):
            if entity in self._scene_objects:
                self._scene_objects.remove(entity)
                entity.scene = None
        elif isinstance(entity, Link):
            if entity in self._links:
                self._links.remove(entity)
        elif isinstance(entity, RobotStructure):
            if entity in self._robot_structures:
                self._robot_structures.remove(entity)
        self.dirty = True

    @property
    def objects(self):
        return tuple(self._scene_objects)

    @property
    def links(self):
        return tuple(self._links)
