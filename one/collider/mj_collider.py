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

    def compile(self, margin=0.0, auto_acm=False):
        self._mjenv = opme.MJEnv(
            self.scene, margin=margin)
        if auto_acm:
            # Allowed Collision Matrix: whatever overlaps in the current (rest)
            # configuration is structural -- compact wrists, neck/torso, a
            # mounted EE sitting on its flange -- so disable those geom pairs by
            # rebuilding the model with them excluded. (MoveIt's "default in
            # collision -> disable" heuristic, here keyed on the rest pose.)
            excludes = self._detect_resting_collisions()
            if excludes:
                self._mjenv = opme.MJEnv(
                    self.scene, margin=margin, extra_excludes=excludes)

    def _detect_resting_collisions(self):
        # Second layer of the ACM. Parent-child (adjacent) links are already
        # excluded structurally (MechStruct.compile -> contact_excludes), so this
        # pass-1 model never reports those contacts -- detection only ever sees
        # NON-adjacent overlaps (compact wrists, neck/torso, a mounted EE on its
        # flange). No overlap with the topological layer, no double handling.
        import mujoco
        env = self._mjenv
        m, d, sync = env.runtime.model, env.runtime.data, env.sync
        # push every mecba's current state, then forward
        for mecba in self.scene.mecbas:
            sync.push_one_mecba_freebase_pose(mecba, mecba.quat, mecba.pos)
            sync.push_one_mecba_qpos(mecba, mecba.qs)
        sync.push_all_sobj_qpos()
        mujoco.mj_forward(m, d)
        # all mecbas incl. mounted children (hands/grippers ride in the model
        # nested under their mount link but aren't in scene.mecbas).
        all_mecbas = []

        def _walk(mb):
            all_mecbas.append(mb)
            for mt in mb._mountings.values():
                if hasattr(mt.child, 'runtime_lnks'):
                    _walk(mt.child)
        for mb in self.scene.mecbas:
            _walk(mb)
        # runtime link -> (mecba, lidx), and mj body id -> runtime link.
        # link bodies live in sync.rutl2bdy (link -> body node), not _body_map
        # (which only holds free SceneObjects).
        lnk2ml = {id(lnk): (mb, i)
                  for mb in all_mecbas
                  for i, lnk in enumerate(mb.runtime_lnks)}
        bid2lnk = {}
        for lnk, body in sync.rutl2bdy.items():
            if id(lnk) in lnk2ml:
                bid2lnk[m.body(body.name).id] = lnk
        pairs, seen = [], set()
        for k in range(d.ncon):
            c = d.contact[k]
            la = bid2lnk.get(m.geom_bodyid[c.geom1])
            lb = bid2lnk.get(m.geom_bodyid[c.geom2])
            if la is None or lb is None or la is lb:
                continue
            key = frozenset((id(la), id(lb)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((lnk2ml[id(la)], lnk2ml[id(lb)]))
        return pairs

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
            # full per-joint width (== n_jnts); push_one_mecba_qpos indexes by
            # full joint id and skips fixed joints. For robots without fixed /
            # mimic joints this equals actor.ndof.
            ndof = len(actor.qs)
            self._actor_qs_slice[actor] = slice(offset, offset + ndof)
            offset += ndof
