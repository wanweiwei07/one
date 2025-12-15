class Scene:

    def __init__(self):
        self._scn_objs = []
        self._dirty = True

    def __iter__(self):
        return iter(self._scn_objs)

    def __getitem__(self, key):
        return self._scn_objs[key]

    def add(self, scn_obj):
        scn_obj.scene = self
        self._scn_objs.append(scn_obj)
        self._dirty = True

    def remove(self, entity):
        self._scn_objs.remove(entity)
        self._dirty = True
