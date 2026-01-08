import one.scene.scene as oss
import one.physics.mj_env as opme


class MjCollider:

    def __init__(self):
        self.scene = oss.Scene()
        self.actor = None
        self._mjenv = None

    def append(self, entity):
        self.scene.add(entity)

    def compile(self):
        self._mjenv = opme.MJEnv(self.scene)

    def is_collided(self, qs):
        if self.actor is None:
            raise RuntimeError("MjCollider.actor is not set!")
        if self._mjenv is None:
            raise RuntimeError("MjCollider must be compiled!")
        self._mjenv.sync.push_by_state(self.actor.state, qs)
        return self._mjenv.is_collided()