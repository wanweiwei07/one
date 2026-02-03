import one.scene.scene as oss
import one.physics.mj_env as opme


class MJCollider:

    def __init__(self):
        self.scene = oss.Scene()
        self._actors = ()
        self._actor_qs_slice = {}
        self._mjenv = None

    def append(self, entity):
        self.scene.add(entity)

    def compile(self, margin=0.0):
        self._mjenv = opme.MJEnv(
            self.scene, margin=margin)

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
            # free base sync (the func has no-op if not free)
            self._mjenv.sync.push_one_mecba_freebase_pose(
                actor, actor.quat, actor.pos)
            # joint qpos sync
            self._mjenv.sync.push_one_mecba_qpos(actor, qs[sl])
        self._mjenv.sync.push_all_sobj_qpos()
        return self._mjenv.runtime.is_collided()

    def get_slice(self, actor):
        return self._actor_qs_slice.get(actor, None)

    def set_mecba_qpos(self, mecba, qs):
        self._mjenv.sync.push_one_mecba_qpos(mecba, qs)

    def save(self, filepath, encoding="utf-8"):
        if self._mjenv is None:
            raise RuntimeError("MjCollider must be compiled!")
        self._mjenv.save(filepath, encoding=encoding)

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
