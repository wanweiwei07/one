import mujoco
import numpy as np
import one.physics.mjcf_utils as mju
import one.physics.mj_contact as mjcv


class Namer:
    def __init__(self):
        self._counter = {}
        self.bdy_names = {}
        self.jnt_names = {}

    def unique_name(self, kind, base):
        key = (kind, base)
        i = self._counter.get(key, 0)
        self._counter[key] = i + 1
        return f"{base}_{i}"

    def reg_bdy(self, sobj, name):
        self.bdy_names[sobj] = name

    def reg_jnt(self, jnt, name):
        self.jnt_names[jnt] = name


class MjEnv:
    def __init__(self, scene, free_root=False):
        self.gravity = np.asarray([0, 0, -9.81], dtype=np.float32)
        self.timestep = 0.002
        self.scene = scene
        self.free_root = free_root
        # muJoCo model/data
        self.model = None
        self.data = None
        # SceneObject -> body_id
        self._body_idx_map = {}
        # state, joint idx -> qpos adr
        self._qpos_by_state_jidx = {}
        # mesh assets
        self._mesh_assets = {}
        # name manager
        self.namer = Namer()
        # xml string cache
        self.xml_string = None
        # build from scene
        self._build_from_scene()
        # collider mode
        self._cd_mode = False  # False: Dynamic, True: Collision Detection
        self._dyn_backup = None  # backup dynamics for cd mode
        # # contact viz
        # self.contact_viz = mjcv.MjContactForceViz(self.scene)

    def step(self, dt):
        self._exit_cd_mode()
        h = self.model.opt.timestep
        substeps = int(round(dt / h))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        self.sync_mujoco_to_mechstates()
        for obj, bid in self._body_idx_map.items():
            mj_rotmat = self.data.xmat[bid].reshape(3, 3)
            mj_pos = self.data.xpos[bid]
            obj.set_rotmat_pos(mj_rotmat, mj_pos)
        # self.contact_viz.update_from_data(self.model, self.data)

    def is_collided(self):
        self._enter_cd_mode()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return self.data.ncon > 0

    def sync_to_mujoco(self, state, qs):
        for jidx, q in enumerate(qs):
            qadr = self._qpos_by_state_jidx[(state, jidx)]
            self.data.qpos[qadr] = q

    def sync_mechstates_to_mujoco(self):
        for state in self.scene.states:
            # print("state.qs =", state.qs)
            qs = state.qs
            for jidx, q in enumerate(qs):
                qadr = self._qpos_by_state_jidx[(state, jidx)]
                print(qadr)
                self.data.qpos[qadr] = q
        #     print("after set qpos:", qs_mj)
        # self.data.qvel[:] = 0
        # self.data.qacc[:] = 0
        # if self.model.na:
        #     self.data.act[:] = 0
        # print("ncon(before step) =", self.data.ncon)
        # mujoco.mj_forward(self.model, self.data)
        # print("ncon(after forward) =", self.data.ncon)
        # TODO use qpos 0-7 for root pose
        # if self.free_root:
        #     pos = state.base_pos
        #     qx, qy, qz, qw = rm.quat_from_rotmat(state.base_rotmat)  # x,y,z,w?
        #     self.data.qpos[0:3] = pos
        #     self.data.qpos[3:7] = (qw, qx, qy, qz)

    def sync_mujoco_to_mechstates(self):
        for state in self.scene.states:
            # TODO only first lnk fixed for now
            if not state.runtime_lnks[0].is_fixed:
                bid = self._body_idx_map[state.runtime_lnks[0]]
                mj_rotmat = self.data.xmat[bid].reshape(3, 3)
                mj_pos = self.data.xpos[bid]
                state.base_rotmat[:] = mj_rotmat
                state.base_pos[:] = mj_pos
            for jidx, _ in enumerate(state.qs):
                qadr = self._qpos_by_state_jidx[(state, jidx)]
                state.qs[jidx] = self.data.qpos[qadr]
            state.fk()
            state.update()

    def save_xml(self, filepath, encoding="utf-8"):
        if self.xml_string is None:
            raise RuntimeError("XML not built yet")
        with open(filepath, "w", encoding=encoding) as f:
            f.write(self.xml_string)

    def _build_from_scene(self):
        assets_xml = []
        bodies_xml = []
        mesh_assets = self._mesh_assets
        for state in self.scene.states:
            assets, root_body = mju.state_to_mjcf_body(state,
                                                       mesh_assets,
                                                       self.namer)
            assets_xml.extend(assets)
            bodies_xml.append(root_body)
        for scn_obj in self.scene.sobjs:
            if not scn_obj.collisions:
                continue
            assets, body = mju.sobj_to_mjcf_body(scn_obj,
                                                 mesh_assets,
                                                 self.namer)
            assets_xml.extend(assets)
            bodies_xml.append(body)
        assets = "\n".join(mju.indent(x, 4) for x in assets_xml)
        bodies = "\n".join(mju.indent(x, 4) for x in bodies_xml)
        self.xml_string = mju.model_template.format(gx=self.gravity[0],
                                                    gy=self.gravity[1],
                                                    gz=self.gravity[2],
                                                    timestep=self.timestep,
                                                    assets=assets,
                                                    bodies=bodies)
        print(self.xml_string)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        self._build_body_map()

    def _build_body_map(self):
        self._body_idx_map.clear()
        # normal objects
        for sobj in self.scene.sobjs:
            if not sobj.collisions:
                continue
            mj_name = self.namer.bdy_names[sobj]
            mb_id = mujoco.mj_name2id(self.model,
                                      mujoco.mjtObj.mjOBJ_BODY,
                                      mj_name)
            assert mb_id >= 0, f"Body name not found in MJCF: {mj_name}"
            self._body_idx_map[sobj] = mb_id
        # structures
        self._qpos_by_state_jidx.clear()
        for state in self.scene.states:
            for lnk in state.runtime_lnks:
                mj_name = self.namer.bdy_names[lnk]
                bid = mujoco.mj_name2id(self.model,
                                        mujoco.mjtObj.mjOBJ_BODY,
                                        mj_name)
                assert bid >= 0
                self._body_idx_map[lnk] = bid
            for jidx, jnt in enumerate(state._compiled._meta.jnts):
                mj_name = self.namer.jnt_names[jnt]
                print(mj_name)
                mj_id = mujoco.mj_name2id(self.model,
                                          mujoco.mjtObj.mjOBJ_JOINT,
                                          mj_name)
                assert mj_id >= 0, f"Joint name not found in MJCF: {mj_name}"
                qadr = self.model.jnt_qposadr[mj_id]
                self._qpos_by_state_jidx[(state, jidx)] = qadr

    def _enter_cd_mode(self):
        if self._cd_mode:
            return
        self._backup_dyn_state()
        self._cd_mode = True

    def _exit_cd_mode(self):
        if not self._cd_mode:
            return
        self._restore_dyn_state()
        self._cd_mode = False

    def _backup_dyn_state(self):
        assert self._dyn_backup is None
        self._dyn_backup = {"qpos": self.data.qpos.copy(),
                            "qvel": self.data.qvel.copy(),
                            "act": self.data.act.copy(),
                            "ctrl": self.data.ctrl.copy(),
                            "time": self.data.time}

    def _restore_dyn_state(self):
        if self._dyn_backup is None:
            return
        b = self._dyn_backup
        self.data.qpos[:] = b["qpos"]
        self.data.qvel[:] = b["qvel"]
        self.data.act[:] = b["act"]
        self.data.ctrl[:] = b["ctrl"]
        self.data.time = b["time"]
        self._dyn_backup = None
        mujoco.mj_forward(self.model, self.data)