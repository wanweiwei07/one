import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.render_model_primitive as osrmp
import one.scene.scene_object_primitive as ossop


class KineVisualizer:
    def __init__(self, mech, mode='chain',
                 axis_length=None, axis_radius=None, link_radius=0.01,
                 stator_rgb=ouc.ExtendedColor.DEEP_GRAY,
                 rotor_rgb=ouc.ExtendedColor.SILVER_GRAY,
                 link_rgb=ouc.ExtendedColor.GOLD2,
                 alpha=1.0):
        if mode not in ('chain', 'structure'):
            raise ValueError("mode must be 'chain' or 'structure'")
        self.mech = mech
        self.mode = mode
        self.link_radius = float(link_radius)
        if axis_radius is None:
            axis_radius = self.link_radius * 1.5
        if axis_length is None:
            axis_length = self.link_radius * 1.2
        self.axis_length = float(axis_length)
        self.axis_radius = float(axis_radius)
        self.stator_rgb = np.asarray(stator_rgb, dtype=np.float32)
        self.rotor_rgb = np.asarray(rotor_rgb, dtype=np.float32)
        self.link_rgb = np.asarray(link_rgb, dtype=np.float32)
        self.alpha = float(alpha)
        self._objects = []

    def _joint_list_and_qs(self):
        if self.mode == 'chain':
            if hasattr(self.mech, '_chain'):
                chain = self.mech._chain
            else:
                compiled = self.mech.structure.compiled
                chain = self.mech.structure.get_chain(compiled.root_lnk, compiled.tip_lnks[0])
            jnts = chain.jnts
            qs = self.mech.qs[chain.jnt_ids_in_structure]
        else:
            jnts = self.mech.structure.jnts
            qs = self.mech.qs
        return jnts, np.asarray(qs, dtype=np.float32)

    @staticmethod
    def _compute_joint_frames(jnts, qs, root_tf):
        origins = []
        axes = []
        tf = root_tf.copy()
        for jnt, q in zip(jnts, qs):
            jtf = tf @ jnt.zero_tf
            origins.append(jtf[:3, 3].copy())
            axes.append((jtf[:3, :3] @ jnt.ax).copy())
            tf = jtf @ jnt.motion_tf(float(q))
        return origins, axes

    def _build_objects(self):
        jnts, qs = self._joint_list_and_qs()
        root_tf = self.mech.tf
        origins, axes = self._compute_joint_frames(jnts, qs, root_tf) if len(jnts) > 0 else ([], [])

        objs = []
        fr_h = self.link_radius * 3.0
        fr_bottom = fr_h
        fr_top = fr_h * (2.0 / 3.0)
        base_pos = self.mech.pos
        base_rotmat = self.mech.rotmat
        base_top = base_pos + base_rotmat[:, 2] * fr_h
        fr_rmodel = osrmp.gen_frustrum_rmodel(
            height=fr_h,
            bottom_length=fr_bottom,
            top_length=fr_top,
            rotmat=base_rotmat,
            pos=base_pos,
            rgb=ouc.ExtendedColor.SALMON_PINK,
            alpha=self.alpha)
        fr_obj = osso.SceneObject(collision_type=None, is_free=False)
        fr_obj.add_visual(fr_rmodel, auto_make_collision=False)
        objs.append(fr_obj)

        if len(origins) > 0:
            objs.append(ossop.dashed_cylinder(
                spos=base_top,
                epos=origins[0],
                radius=self.link_radius,
                rgb=ouc.ExtendedColor.SALMON_PINK,
                alpha=self.alpha))

        for origin, axis in zip(origins, axes):
            axis_u = oum.unit_vec(axis, return_length=False)
            # Stator is on negative axis side, rotor on positive axis side.
            stator_s = origin - axis_u * self.axis_length
            stator_e = origin
            rotor_s = origin
            rotor_e = origin + axis_u * self.axis_length
            objs.append(ossop.cylinder(
                spos=stator_s, epos=stator_e, radius=self.axis_radius,
                rgb=self.stator_rgb, alpha=self.alpha))
            objs.append(ossop.cylinder(
                spos=rotor_s, epos=rotor_e, radius=self.axis_radius,
                rgb=self.rotor_rgb, alpha=self.alpha))

        for i in range(len(origins) - 1):
            if np.linalg.norm(origins[i + 1] - origins[i]) < 1e-8:
                continue
            objs.append(ossop.cylinder(
                spos=origins[i], epos=origins[i + 1], radius=self.link_radius,
                rgb=self.link_rgb, alpha=self.alpha))
        return objs

    def attach_to(self, scene):
        if not self._objects:
            self._objects = self._build_objects()
        for obj in self._objects:
            scene.add(obj)

    def detach_from(self, scene):
        for obj in self._objects:
            scene.remove(obj)
        self._objects = []

    def refresh(self, scene):
        self.detach_from(scene)
        self.attach_to(scene)
