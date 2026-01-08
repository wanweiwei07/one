import mujoco

class MJSynchronizer:
    def __init__(self, mj_runtime, scene):
        self.mj_runtime = mj_runtime
        self.scene = scene
        self._body_map = {}
        self._qpos_map = {}
        self._build_maps()

    def _build_maps(self):
        model = self.mj_runtime.model
        for obj in self.scene.sobjs:
            if not obj.collisions:
                continue
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, obj.name)
            self._body_map[obj] = bid
        for state in self.scene.states:
            compiled = state._compiled
            for jidx in range(compiled.n_jnts):
                lidx = compiled.clidx_of_jidx[jidx]
                lnk = state.runtime_lnks[lidx]
                jname = f"j{jidx}({lnk.name})"
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                qadr = model.jnt_qposadr[jid]
                self._qpos_map[(state, jidx)] = qadr

    def push_qpos(self):
        data = self.mj_runtime.data
        for (state, jidx), adr in self._qpos_map.items():
            data.qpos[adr] = state.qs[jidx]

    def pull_qpos(self):
        data = self.mj_runtime.data
        for (state, jidx), adr in self._qpos_map.items():
            state.qs[jidx] = data.qpos[adr]
        for state in self.scene.states:
            state.fk()
            state.update()

    def pull_body_pose(self):
        data = self.mj_runtime.data
        for obj, bid in self._body_map.items():
            pos = data.xpos[bid]
            rot = data.xmat[bid].reshape(3,3)
            obj.set_rotmat_pos(rot, pos)
