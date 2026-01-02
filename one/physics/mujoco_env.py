import mujoco
import numpy as np
import one.utils.math as rm
import one.scene.collision as sco
import one.physics.mjcf_utils as mju


class MuJoCoEnv:
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
        # Compiled structure joint idx -> qpos adr
        self._compiled_to_qpos = {}
        # mesh assets bookkeeping
        self._mesh_assets = {}
        self._mesh_count = 0
        # xml string cache
        self.xml_string = None
        # build from scene
        self._build_from_scene()

    def step(self, dt):
        h = self.model.opt.timestep
        substeps = int(round(dt / h))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        self.sync_mujoco_to_mechstates()
        for obj, bid in self._body_idx_map.items():
            mj_rotmat = self.data.xmat[bid].reshape(3, 3)
            mj_pos = self.data.xpos[bid]
            obj.set_rotmat_pos(mj_rotmat, mj_pos)

    def sync_mechstates_to_mujoco(self):
        for state in self.scene.states:
            # print("state.qs =", state.qs)
            qs = state.qs
            qs_mj = []
            for jidx, q in enumerate(qs):
                qadr = self._compiled_to_qpos[(state, jidx)]
                self.data.qpos[qadr] = q
                qs_mj.append(self.data.qpos[qadr])
        #     print("after set qpos:", qs_mj)
        # self.data.qvel[:] = 0
        # self.data.qacc[:] = 0
        # if self.model.na:
        #     self.data.act[:] = 0
        print("ncon(before step) =", self.data.ncon)
        # mujoco.mj_forward(self.model, self.data)
        print("ncon(after forward) =", self.data.ncon)
        # TODO use qpos 0-7 for root pose
            # if self.free_root:
            #     pos = state.base_pos
            #     qx, qy, qz, qw = rm.quat_from_rotmat(state.base_rotmat)  # x,y,z,w?
            #     self.data.qpos[0:3] = pos
            #     self.data.qpos[3:7] = (qw, qx, qy, qz)

    def sync_mujoco_to_mechstates(self):
        for state in self.scene.states:
            # if self.free_root:
            #     pos = self.data.qpos[0:3]
            #     w, x, y, z = self.data.qpos[3:7]
            #     rotmat = rm.rotmat_from_quat([x, y, z, w])
            #     state.base_pos[:] = pos
            #     state.base_rotmat[:] = rotmat
            for jidx, _ in enumerate(state.qs):
                qadr = self._compiled_to_qpos[(state, jidx)]
                state.qs[jidx] = self.data.qpos[qadr]
            print(state.qs)
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
        mesh_counter = [self._mesh_count]
        for state in self.scene.states:
            assets, root_body = mju.state_to_mjcf_body(state,
                                                       mesh_assets,
                                                       mesh_counter,
                                                       root_freejoint=self.free_root)
            assets_xml.extend(assets)
            bodies_xml.append(root_body)
            self._mesh_count = mesh_counter[0]
        for scn_obj in self.scene.sobjs:
            if not scn_obj.collisions:
                continue
            assets, body = mju.sceneobject_to_mjcf_body(scn_obj,
                                                        mesh_assets,
                                                        mesh_counter,
                                                        freejoint=True)
            assets_xml.extend(assets)
            bodies_xml.append(body)
        self.xml_string = mju.model_template.format(gx=self.gravity[0], gy=self.gravity[1], gz=self.gravity[2],
                                                    timestep=self.timestep,
                                                    assets="\n".join(mju.indent(x, 4) for x in assets_xml),
                                                    bodies="\n".join(mju.indent(x, 4) for x in bodies_xml))
        print(self.xml_string)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        self._build_body_map()

        print("nu(actuators) =", self.model.nu)
        print("na(state)     =", self.model.na)
        print("nv(dofs)      =", self.model.nv)

        if self.model.nu:
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                print("act", i, name,
                      "trntype", int(self.model.actuator_trntype[i]),
                      "biastype", int(self.model.actuator_biastype[i]))
        print("jnt_stiffness:", self.model.jnt_stiffness)  # 只要这里有非零，就会“回参考位”
        print("dof_damping:", self.model.dof_damping)  # damping 不会在 qvel=0 时产生力

    def _build_body_map(self):
        self._body_idx_map.clear()
        # normal objects
        for sobj in self.scene.sobjs:
            if not sobj.collisions:
                continue
            mb_id = mujoco.mj_name2id(self.model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    sobj.name)
            assert mb_id >= 0, f"Body name not found in MJCF: {sobj.name}"
            self._body_idx_map[sobj] = mb_id
        # structures
        self._compiled_to_qpos.clear()
        for state in self.scene.states:
            for jidx, jnt in enumerate(state._compiled._meta.jnts):
                mj_id = mujoco.mj_name2id(self.model,
                                          mujoco.mjtObj.mjOBJ_JOINT,
                                          jnt.name)
                assert mj_id >= 0, f"Joint name not found in MJCF: {jnt.name}"
                qadr = self.model.jnt_qposadr[mj_id]
                self._compiled_to_qpos[(state, jidx)] = qadr
