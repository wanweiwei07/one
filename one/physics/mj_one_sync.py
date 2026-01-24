class MJSynchronizer:
    def __init__(self, mj_runtime, scene,
                 sobj2bdy, rutl2bdy, mecj2jnt):
        self.mj_runtime = mj_runtime
        self.scene = scene
        self.sobj2bdy = sobj2bdy
        self.rutl2bdy = rutl2bdy
        self.mecj2jnt = mecj2jnt
        self._body_map = {}
        self._qpos_map = {}
        self._freebase_map = {}
        self._freebase_qpos_adr = {}
        self._sobj_qpos_adr = {}
        self._build_maps()

    def _build_maps(self):
        model = self.mj_runtime.model
        for sobj, body in self.sobj2bdy.items():
            bid = self.mj_runtime.model.body(body.name).id
            self._body_map[sobj] = bid
            if not sobj.is_free:
                continue # only free bodies have qpos
            jid = model.body_jntadr[bid]
            qadr = model.jnt_qposadr[jid]
            self._sobj_qpos_adr[sobj] = qadr
        for mecba in self.scene.mecbas:
            root_lnk = mecba.runtime_root_lnk
            if root_lnk.is_free:
                body = self.rutl2bdy[root_lnk]
                bid = model.body(body.name).id
                self._freebase_map[mecba] = bid
                jid = model.body_jntadr[bid]
                qadr = model.jnt_qposadr[jid]
                self._freebase_qpos_adr[mecba] = qadr
        for (mecba, jidx), jnode in self.mecj2jnt.items():
            jid = self.mj_runtime.model.joint(jnode.name).id
            qadr = model.jnt_qposadr[jid]
            self._qpos_map[(mecba, jidx)] = qadr

    def pull_all_mecba_qpos(self):
        data = self.mj_runtime.data
        for (mecba, jidx), adr in self._qpos_map.items():
            mecba.qs[jidx] = data.qpos[adr]
        for mecba in self.scene.mecbas:
            mecba.fk()

    def pull_all_mecba_freebase_pose(self):
        data = self.mj_runtime.data
        for mecba, bid in self._freebase_map.items():
            rotmat = data.xmat[bid].reshape(3, 3)
            pos = data.xpos[bid]
            mecba.set_rotmat_pos(rotmat, pos)

    def pull_all_sobj_pose(self):
        data = self.mj_runtime.data
        for sobj, bid in self._body_map.items():
            pos = data.xpos[bid]
            rot = data.xmat[bid].reshape(3, 3)
            sobj.set_rotmat_pos(rot, pos)

    def push_all_sobj_qpos(self):
        data = self.mj_runtime.data
        for sobj, qadr in self._sobj_qpos_adr.items():
            pos = sobj.pos
            quat = sobj.quat  # [x, y, z, w]
            data.qpos[qadr:qadr + 3] = pos
            qx, qy, qz, qw = quat
            data.qpos[qadr + 3:qadr + 7] = [qw, qx, qy, qz]

    def push_one_sobj_qpos(self, sobj, quat, pos):
        if sobj not in self._sobj_qpos_adr:
            return
        qadr = self._sobj_qpos_adr[sobj]
        data = self.mj_runtime.data
        data.qpos[qadr:qadr + 3] = pos
        qx, qy, qz, qw = quat
        data.qpos[qadr + 3:qadr + 7] = [qw, qx, qy, qz]

    def push_all_mecba_qpos(self):
        data = self.mj_runtime.data
        for (state, jidx), adr in self._qpos_map.items():
            data.qpos[adr] = state.qs[jidx]

    def push_one_mecba_qpos(self, mecba, qs):
        data = self.mj_runtime.data
        for jidx, q in enumerate(qs):
            qadr = self._qpos_map[(mecba, jidx)]
            data.qpos[qadr] = q

    def push_one_mecba_freebase_pose(self, mecba, quat, pos):
        """
        Only for free base. Writes free joint qpos (pos + quat) then mj_forward.
        """
        if mecba not in self._freebase_qpos_adr:
            return
        qadr = self._freebase_qpos_adr[mecba]
        data = self.mj_runtime.data
        data.qpos[qadr:qadr + 3] = pos
        qx, qy, qz, qw = quat
        data.qpos[qadr + 3:qadr + 7] = [qw, qx, qy, qz]
