import one.scene.geometry as geom
import one.scene.model as mdl


class Scene:

    def __init__(self):
        self._models = []
        self._geoms = []
        self._dirty_set = set()

    def __iter__(self):
        return iter(self._models+self._geoms)

    def __getitem__(self, key):
        return self._models[key]

    def add(self, model):
        if isinstance(model, geom.GeometryBase):
            self._geoms.append(model)
        if isinstance(model, mdl.Model):
            self._models.append(model)
            model.set_parent(None)

    def remove(self, model):
        self._models.remove(model)
