import one.scene.scene as oss
import one.physics.mj_env as opme


class MjCollider:

    def __init__(self):
        self.scene = oss.Scene()
        self._actors = ()
        self._actor_qs_slice = {}
        self._mjenv = None

    def append(self, entity):
        self.scene.add(entity)

    def compile(self):
        self._mjenv = opme.MJEnv(self.scene)

    def is_collided(self, qs):
        if not self.actors:
            raise RuntimeError("MjCollider.actor is not set!")
        if self._mjenv is None:
            raise RuntimeError("MjCollider must be compiled!")
        for actor, sl in self._actor_qs_slice.items():
            if actor not in self.scene.mecbas:
                raise RuntimeError(
                    "All MjCollider.actors must be"
                    " added to the scene!")
            self._mjenv.sync.push_by_mecba(actor, qs[sl])
        return self._mjenv.is_collided()

    def get_slice(self, actor):
        return self._actor_qs_slice.get(actor, None)

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, actors):
        if not actors:
            raise ValueError("MjCollider.actors cannot be empty!")
        self._actors = tuple(actors)
        self._rebuild_mapping()

    def _rebuild_mapping(self):
        self._actor_qs_slice.clear()
        offset = 0
        for actor in self._actors:
            if actor not in self.scene.mecbas:
                raise RuntimeError(
                    "All MjCollider.actors must be added to the scene!")
            ndof = actor.ndof
            self._actor_qs_slice[actor] = slice(offset, offset + ndof)
            offset += ndof
