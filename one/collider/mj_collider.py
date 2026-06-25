import one.scene.scene as oss
import one.physics.mj_env as opme


class MJCollider:
    """The robot's collision world: the planned mechanism(s) (``actors``), the
    static ``fixtures``, and the movable ``objects`` they manipulate, compiled to
    a MuJoCo model for collision queries.

    Built ONCE and reused across many motion queries; only a STRUCTURAL change
    (an object grasped / released / excluded) triggers a ``refresh`` (rebuild),
    which reuses the cached self-collision matrix instead of re-detecting it.

    Two collision exemptions are derived AUTOMATICALLY on (re)build, so callers
    never hand-roll them:
      * a grasped object (a plain SceneObject mounted on an actor's end effector)
        is exempted from that end effector's links -- the intended grasp contact;
      * an ``exclude``d object (e.g. the grasp target during its own approach) is
        exempted from all actor links.
    Grasping itself stays on the end effector (``ee.attach`` / ``ee.detach``);
    this class only reflects it -- there is no grasp/hold verb here. Call
    ``refresh`` after an ``attach`` / ``detach`` / ``exclude`` / ``include`` to
    apply the change.
    """

    def __init__(self, robot=None, fixtures=(), objects=()):
        self.scene = oss.Scene()
        self._actors = ()
        self._actor_qs_slice = {}
        self._mjenv = None
        self.acm = ()               # cached self-collision excludes (link pairs)
        self._objects = []          # movable objects (tracked for exclude/include)
        self._excluded = set()      # objects currently NOT treated as obstacles
        self._margin = 0.0
        if robot is not None:
            self.append(robot)
            self.actors = [robot]
        for f in fixtures:
            self.append(f)
        for o in objects:
            self.append(o)
            self._objects.append(o)

    def append(self, entity):
        self.scene.add(entity)

    # ---- (re)build -----------------------------------------------------------
    def compile(self, margin=0.0, auto_acm=True):
        """Build the MuJoCo model. With ``auto_acm`` DETECT the resting (rest-pose
        structural) self-collisions once and CACHE them on ``self.acm`` so later
        ``refresh`` calls reuse them. Held-object / excluded-object exemptions are
        derived automatically (see class docstring)."""
        self._margin = margin
        self.acm = ()
        self._build(detect_acm=auto_acm)

    def refresh(self, margin=None):
        """Rebuild after a structural change (grasp / release / exclude), REUSING
        the cached ``acm`` (no re-detection) and re-deriving the automatic
        exemptions. Cheaper than ``compile`` and what callers use between phases."""
        if margin is not None:
            self._margin = margin
        self._build(detect_acm=False)

    def _build(self, detect_acm):
        excludes = self._auto_excludes()
        self._mjenv = opme.MJEnv(
            self.scene, margin=self._margin, extra_excludes=excludes or None)
        if detect_acm:
            self.acm = tuple(self._detect_resting_collisions())
        if self.acm:
            self._mjenv = opme.MJEnv(
                self.scene, margin=self._margin,
                extra_excludes=excludes + list(self.acm))

    # ---- automatic exemptions ------------------------------------------------
    def _mounted_mechs(self):
        """Every actor mech plus its mounted mech children (grippers / hands)."""
        out = []

        def _walk(mb):
            out.append(mb)
            for mt in mb._mountings.values():
                if hasattr(mt.child, 'runtime_lnks'):
                    _walk(mt.child)
        for a in self._actors:
            _walk(a)
        return out

    def _auto_excludes(self):
        """(a) grasped object vs its carrier-mech links; (b) excluded object vs
        all actor links."""
        pairs = []
        mechs = self._mounted_mechs()
        # (a) held objects: a plain SceneObject mounted on a mech -> exempt vs
        # that mech's links (the intended grasp contact).
        for mb in mechs:
            for mt in mb._mountings.values():
                child = mt.child
                if not hasattr(child, 'runtime_lnks'):
                    for lnk in mb.runtime_lnks:
                        pairs.append((child, lnk))
        # (b) excluded objects (e.g. the grasp target during its approach) ->
        # exempt vs every actor link so the reaching hand may overlap them.
        if self._excluded:
            all_lnks = [lnk for mb in mechs for lnk in mb.runtime_lnks]
            for obj in self._excluded:
                for lnk in all_lnks:
                    pairs.append((obj, lnk))
        return pairs

    # ---- movable-object obstacle role ---------------------------------------
    def exclude(self, obj):
        """Stop treating ``obj`` as an obstacle (e.g. the grasp target during its
        own approach -- the grasp set guarantees clearance). Apply with
        ``refresh``."""
        self._excluded.add(obj)

    def include(self, obj):
        """Treat ``obj`` as an obstacle again. Apply with ``refresh``."""
        self._excluded.discard(obj)

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
        all_mecbas = self._mounted_mechs()
        # runtime link -> (mecba, lidx), and mj body id -> runtime link.
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
            pairs.append((la, lb))
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
            ndof = len(actor.qs)
            self._actor_qs_slice[actor] = slice(offset, offset + ndof)
            offset += ndof
