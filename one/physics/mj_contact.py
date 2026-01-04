import mujoco
import numpy as np
import one.utils.math as rm
import one.utils.constant as const
import one.scene.scene_object_primitive as prims


class MjContactViz:
    def __init__(self, scene, max_contacts=64, radius=0.01):
        self.scene = scene
        self.max_contacts = max_contacts
        self._spheres = []
        self._init_spheres(radius)

    def _init_spheres(self, radius):
        for _ in range(self.max_contacts):
            s = prims.gen_sphere(radius=radius,
                                 rgb=(1, 0, 0),
                                 alpha=const.ALPHA.SEMI,
                                 collision_type=None,
                                 is_fixed=True)
            s.attach_to(self.scene)
            self._spheres.append(s)

    def clear(self):
        for s in self._spheres:
            s.alpha = const.ALPHA.TRANSPARENT

    def update_from_data(self, mj_data):
        nc = min(mj_data.ncon, self.max_contacts)
        self.clear()
        for i in range(nc):
            cp = mj_data.contact[i]
            pos = cp.pos
            s = self._spheres[i]
            s.pos = pos
            s.alpha = const.ALPHA.NEAR_SOLID


class MjContactForceViz:
    def __init__(self, scene,
                 max_contacts=64,
                 base_length=const.ForceArrowSize.BASE_LENGTH,
                 gain=const.ForceArrowSize.GAIN):
        self.scene = scene
        self.max_contacts = max_contacts
        self.base_length = base_length
        self.gain = gain
        self._arrows = []
        self._init_arrows()

    def _init_arrows(self):
        for _ in range(self.max_contacts):
            a = prims.gen_arrow(spos=np.zeros(3),
                                epos=np.array([0.0, 0.0, self.base_length]),
                                shaft_radius=const.ForceArrowSize.SHAFT_RADIUS,
                                head_radius=const.ForceArrowSize.HEAD_RADIUS,
                                head_length=const.ForceArrowSize.HEAD_LENGTH,
                                segments=8,
                                rgb=(1, 0, 0),
                                alpha=const.ALPHA.SEMI,
                                collision_type=None,
                                is_fixed=True)
            a.attach_to(self.scene)
            self._arrows.append(a)

    def clear(self):
        for a in self._arrows:
            a.alpha = const.ALPHA.TRANSPARENT

    def update_from_data(self, mj_model, mj_data):
        self.clear()
        nc = min(mj_data.ncon, self.max_contacts)
        force = np.zeros(6)
        for i in range(nc):
            cp = mj_data.contact[i]
            spos = cp.pos
            mujoco.mj_contactForce(mj_model, mj_data, i, force)
            fn_vec =  -cp.frame.reshape(3,3) @ force[:3]
            fn = np.linalg.norm(fn_vec)
            if fn < 1e-10:
                continue
            direction = fn_vec / fn
            rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, direction)
            strength = np.tanh(fn * self.gain)
            r = strength
            g = strength * (1 - strength)
            b = 1 - strength
            rgba = (r, g, b, const.ALPHA.SOLID)
            a = self._arrows[i]
            a.set_rotmat_pos(rotmat, pos=spos)
            a.rgba = rgba