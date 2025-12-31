import mujoco
import one.scene.collision as sco


class MuJoCoEnv:
    def __init__(self, scene, freejoint_default=True):
        self.scene = scene
        self.freejoint_default = freejoint_default
        # muJoCo model/data
        self.model = None
        self.data = None
        # SceneObject -> body_id
        self._body_map = {}
        # xml asset section
        self._assets_xml = []
        self._bodies_xml = []
        # mesh assets bookkeeping
        self._mesh_assets = {}
        self._mesh_count = 0
        # build from scene
        self._build_from_scene()

    def step(self, dt):
        h = self.model.opt.timestep
        substeps = int(round(dt / h))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        # import numpy as np
        # Rfix = np.array([
        #     [0, -1, 0],
        #     [1, 0, 0],
        #     [0, 0, 1]], dtype=np.float32)
        for obj, bid in self._body_map.items():
            print(obj.name, "pos:", self.data.xpos[bid], "rotmat:", self.data.xmat[bid].reshape(3, 3))
            rotmat = self.data.xmat[bid].reshape(3, 3)
            pos = self.data.xpos[bid]
            obj.set_rotmat_pos(rotmat, pos)

    def _build_from_scene(self):
        self._assets_xml.clear()
        self._bodies_xml.clear()
        self._body_map.clear()
        for scn_obj in self.scene._scn_objs:
            if not scn_obj.collisions:
                continue
            assets, body = self._sceneobj_to_mjcf(scn_obj)
            if assets:
                self._assets_xml.append(assets)
            self._bodies_xml.append(body)
        xml = f"""
                <mujoco>
                  <option gravity="0 0 0"/>
                  <option timestep="0.002"/>
                  <asset>
                    {' '.join(self._assets_xml)}
                  </asset>
                  <worldbody>
                    {' '.join(self._bodies_xml)}
                  </worldbody>
                </mujoco>
                """
        print(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        for sobj in self.scene:
            bid = mujoco.mj_name2id(self.model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    sobj.name)
            self._body_map[sobj] = bid

    def _sceneobj_to_mjcf(self, scn_obj):
        name = scn_obj.name
        pos = scn_obj.pos
        qw, qx, qy, qz = scn_obj.quat
        body_attr = (f'pos="{pos[0]} {pos[1]} {pos[2]}" '
                     f'quat="{qw} {qx} {qy} {qz}"')
        # inertial
        inertial_xml = ""
        if scn_obj.mass is not None:
            com = scn_obj.com if scn_obj.com is not None else (0, 0, 0)
            if scn_obj.inertia is not None:
                I = scn_obj.inertia
                inertial_xml = (f'<inertial mass="{scn_obj.mass}" '
                                f'pos="{com[0]} {com[1]} {com[2]}" '
                                f'fullinertia="{I[0][0]} {I[1][1]} {I[2][2]} '
                                f'{I[0][1]} {I[0][2]} {I[1][2]}"/>')
            else:
                inertial_xml = (f'<inertial mass="{scn_obj.mass}" '
                                f'pos="{com[0]} {com[1]} {com[2]}"/>')
        # geoms
        geoms_xml = []
        assets_xml = []
        for c in scn_obj.collisions:
            geom_xml, asset_xml = self._collision_to_mjcf(c)
            geoms_xml.append(geom_xml)
            if asset_xml:
                assets_xml.append(asset_xml)
        geoms_str = "\n".join(geoms_xml)
        assets_str = "\n".join(assets_xml)
        joint_xml = ""
        if self.freejoint_default and not isinstance(c, sco.PlaneCollisionShape):
            joint_xml = "<freejoint/>"
        body_xml = f"""
                    <body name="{name}" {body_attr}>
                      {joint_xml}
                      {inertial_xml}
                      {geoms_str}
                    </body>
                    """
        return assets_str, body_xml

    def _collision_to_mjcf(self, c):
        pos = c.pos
        qx, qy, qz, qw = c.quat
        common = f'pos="{pos[0]} {pos[1]} {pos[2]}" quat="{qw} {qx} {qy} {qz}"'
        asset_xml = None
        if isinstance(c, sco.SphereCollisionShape):
            geom_xml = f'<geom type="sphere" size="{c.radius}" {common}/>'
            return geom_xml, asset_xml
        elif isinstance(c, sco.CapsuleCollisionShape):
            r, l = c.radius, c.half_length
            geom_xml = f'<geom type="capsule" size="{r} {l}" {common}/>'
            return geom_xml, None
        elif isinstance(c, (sco.AABBCollisionShape, sco.OBBCollisionShape)):
            sx, sy, sz = c.half_extents
            geom_xml = f'<geom type="box" size="{sx} {sy} {sz}" {common}/>'
            return geom_xml, asset_xml
        elif isinstance(c, sco.PlaneCollisionShape):
            geom_xml = f'<geom type="plane"  size="1 1 0.1" {common}/>'
            return geom_xml, asset_xml
        elif isinstance(c, sco.MeshCollisionShape):
            if c.file_path not in self._mesh_assets:
                mesh_name = f"mesh_{self._mesh_count}"
                self._mesh_assets[c.file_path] = mesh_name
                self._mesh_count += 1
                asset_xml = f'<mesh name="{mesh_name}" file="{c.file_path}"/>'
            else:
                mesh_name = self._mesh_assets[c.file_path]
            geom_xml = f'<geom type="mesh" mesh="{mesh_name}" {common}/>'
            return geom_xml, asset_xml
        else:
            raise NotImplementedError(f"Unsupported collision type: {type(c)}")
