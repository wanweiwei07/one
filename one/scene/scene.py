class Scene:

    def __init__(self):
        self._models = []
        self._dirty_set = set()

    def __iter__(self):
        return iter(self._models)

    def __getitem__(self, key):
        return self._models[key]

    def add(self, model):
        self._models.append(model)
        model.set_parent(None)

    def remove(self, model):
        self._models.remove(model)