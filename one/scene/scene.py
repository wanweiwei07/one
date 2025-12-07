class Scene:

    def __init__(self):
        self._entities = []
        self._dirty_set = set()

    def __iter__(self):
        return iter(self._entities)

    def __getitem__(self, key):
        return self._entities[key]

    def add(self, entity):
        self._entities.append(entity)

    def remove(self, entity):
        self._entities.remove(entity)
