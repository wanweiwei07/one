import one.scene.scene as oss
import one.physics.mj_env as opme


class MjCollider:

    def __init__(self):
        self.scene = oss.Scene()
        self.actors = []
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
        for actor in self.actors:
            if actor not in self.scene.mecbas:
                raise RuntimeError(
                    "All MjCollider.actors must be"
                    " added to the scene!")
            self._mjenv.sync.push_by_mecba(actor, qs)
        return self._mjenv.is_collided()