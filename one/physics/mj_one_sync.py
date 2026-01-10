class MJSynchronizer:
    def __init__(self, mj_runtime, scene,
                 sobj2bdy, lnk2bdy, mecj2jnt):
        self.mj_runtime = mj_runtime
        self.scene = scene
        self.sobj2bdy = sobj2bdy
        self.lnk2bdy = lnk2bdy
        self.mecj2jnt = mecj2jnt
        self._body_map = {}
        self._qpos_map = {}
        self._freebase_map = {}
        self._build_maps()

    def _build_maps(self):
        model = self.mj_runtime.model
        for obj, body in self.sobj2bdy.items():
            bid = self.mj_runtime.model.body(body.name).id
            self._body_map[obj] = bid
        for mecba in self.scene.mecbas:
            root_lnk = mecba.structure.compiled.root_lnk
            if root_lnk.is_free:
                body = self.lnk2bdy[(mecba, root_lnk)]
                bid = model.body(body.name).id
                self._freebase_map[mecba] = bid
        for (mecba, jidx), jnode in self.mecj2jnt.items():
            jid = self.mj_runtime.model.joint(jnode.name).id
            qadr = model.jnt_qposadr[jid]
            self._qpos_map[(mecba, jidx)] = qadr

    def push_qpos(self):
        data = self.mj_runtime.data
        for (state, jidx), adr in self._qpos_map.items():
            data.qpos[adr] = state.qs[jidx]

    def pull_qpos(self):
        data = self.mj_runtime.data
        for mecba, bid in self._freebase_map.items():
            pos = data.xpos[bid]
            rot = data.xmat[bid].reshape(3, 3)
            mecba.base_rotmat[:] = rot
            mecba.base_pos[:] = pos
        for (mecba, jidx), adr in self._qpos_map.items():
            mecba.qs[jidx] = data.qpos[adr]
        for mecba in self.scene.mecbas:
            mecba.fk(update=True)

    def pull_body_pose(self):
        data = self.mj_runtime.data
        for obj, bid in self._body_map.items():
            pos = data.xpos[bid]
            rot = data.xmat[bid].reshape(3, 3)
            obj.set_rotmat_pos(rot, pos)

    def push_by_mecba(self, mecba, qs):
        data = self.mj_runtime.data
        for jidx, q in enumerate(qs):
            qadr = self._qpos_map[(mecba, jidx)]
            data.qpos[qadr] = q
