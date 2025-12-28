import mujoco
import numpy as np


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
        self._build_from_scene()

    def _build_from_scene(self):
        self._assets_xml.clear()
        self._bodies_xml.clear()
        self._body_map.clear()
        for scn_obj in self.scene._scn_objs:
            assets, body = self._sceneobj_to_mjcf(scn_obj)
            if assets:
                self._assets_xml.append(assets)
            self._bodies_xml.append(body)
        xml = f"""
                <mujoco>
                  <option gravity="0 0 -9.81"/>
                  <asset>
                    {' '.join(self._assets_xml)}
                  </asset>
                  <worldbody>
                    {' '.join(self._bodies_xml)}
                  </worldbody>
                </mujoco>
                """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        for obj in self.scene.objects:
            bid = mujoco.mj_name2id(self.model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    obj.name)
            self._body_map[obj] = bid

    def _geom_from_model(self, model, obj_name):
        g = model.geometry
        if g.type == "box":
            sx, sy, sz = g.size
            return f'<geom type="box" size="{sx} {sy} {sz}"/>', None
        elif g.type == "sphere":
            return f'<geom type="sphere" size="{g.radius}"/>', None
        elif g.type == "capsule":
            return f'<geom type="capsule" size="{g.radius} {g.half_length}"/>', None
        elif g.type == "mesh":
            mesh_name = f"{obj_name}_mesh"
            return (
                f'<geom type="mesh" mesh="{mesh_name}"/>',
                f'<mesh name="{mesh_name}" file="{g.filepath}"/>'
            )
        else:
            raise NotImplementedError(f"unsupported geom type: {g.type}")

    def _sceneobj_to_mjcf(self, scn_obj):
        name = scn_obj.name
        pos = scn_obj.pos
        qw, qx, qy, qz = scn_obj.quat
        body_attr = (
            f'pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'quat="{qw} {qx} {qy} {qz}"'
        )
        # inertial
        inertial_xml = ""
        if scn_obj.mass is not None:
            com = scn_obj.com if scn_obj.com is not None else (0, 0, 0)
            if scn_obj.inertia is not None:
                I = scn_obj.inertia
                inertial_xml = (
                    f'<inertial mass="{scn_obj.mass}" pos="{com[0]} {com[1]} {com[2]}" '
                    f'fullinertia="{I[0]} {I[1]} {I[2]} {I[3]} {I[4]} {I[5]}"/>'
                )
            else:
                inertial_xml = (
                    f'<inertial mass="{scn_obj.mass}" pos="{com[0]} {com[1]} {com[2]}"/>'
                )
        # geoms
        geoms_xml = []
        assets_xml = []
        models = scn_obj.collisions if scn_obj.collisions else scn_obj.visuals
        for m in models:
            geom_xml, asset_xml = self._geom_from_model(m, name)
            geoms_xml.append(geom_xml)
            if asset_xml:
                assets_xml.append(asset_xml)
        geoms_str = "\n".join(geoms_xml)
        assets_str = "\n".join(assets_xml)
        joint_xml = "<freejoint/>" if self.freejoint_default else ""
        body_xml = f"""
                    <body name="{name}" {body_attr}>
                      {joint_xml}
                      {inertial_xml}
                      {geoms_str}
                    </body>
                    """
        return assets_str, body_xml

    def step(self, substeps=1):
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        for obj, bid in self._body_map.items():
            pos = self.data.xpos[bid]
            rot = self.data.xmat[bid].reshape(3, 3)
            obj.set_rotmat_pos(rot, pos)
