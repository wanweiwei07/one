import mujoco
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop


def debug_contacts(mjenv):
    print("ncon =", mjenv.data.ncon)
    for i in range(mjenv.data.ncon):
        c = mjenv.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = mjenv.model.geom_bodyid[g1]
        b2 = mjenv.model.geom_bodyid[g2]
        body1 = mujoco.mj_id2name(mjenv.model, mujoco.mjtObj.mjOBJ_BODY, b1)
        body2 = mujoco.mj_id2name(mjenv.model, mujoco.mjtObj.mjOBJ_BODY, b2)
        print(i, body1, "<->", body2, "dist=", c.dist)


class MJContactViz:
    def __init__(self, scene, max_contacts=64, radius=0.01):
        self.scene = scene
        self.max_contacts = max_contacts
        self._spheres = []
        self._init_spheres(radius)

    def _init_spheres(self, radius):
        for _ in range(self.max_contacts):
            s = ossop.gen_sphere(radius=radius,
                                 rgb=(1, 0, 0),
                                 alpha=ouc.ALPHA.SEMI,
                                 collision_type=None,
                                 is_fixed=True)
            s.attach_to(self.scene)
            self._spheres.append(s)

    def clear(self):
        for s in self._spheres:
            s.alpha = ouc.ALPHA.TRANSPARENT

    def update_from_data(self, mj_data):
        nc = min(mj_data.ncon, self.max_contacts)
        self.clear()
        for i in range(nc):
            cp = mj_data.contact[i]
            pos = cp.pos
            s = self._spheres[i]
            s.pos = pos
            s.alpha = ouc.ALPHA.NEAR_SOLID


class MjContactForceViz:
    def __init__(self, scene,
                 max_contacts=64,
                 base_length=ouc.ForceArrowSize.BASE_LENGTH,
                 gain=ouc.ForceArrowSize.GAIN):
        self.scene = scene
        self.max_contacts = max_contacts
        self.base_length = base_length
        self.gain = gain
        self._arrows = []
        self._init_arrows()

    def _init_arrows(self):
        for _ in range(self.max_contacts):
            a = ossop.gen_arrow(
                spos=np.zeros(3),
                epos=np.array([0.0, 0.0, self.base_length]),
                shaft_radius=ouc.ForceArrowSize.SHAFT_RADIUS,
                head_radius=ouc.ForceArrowSize.HEAD_RADIUS,
                head_length=ouc.ForceArrowSize.HEAD_LENGTH,
                segments=8, rgb=(1, 0, 0),
                alpha=ouc.ALPHA.SEMI,
                collision_type=None,
                is_fixed=True)
            a.attach_to(self.scene)
            self._arrows.append(a)

    def clear(self):
        for a in self._arrows:
            a.alpha = ouc.ALPHA.TRANSPARENT

    def update_from_data(self, mj_model, mj_data):
        self.clear()
        nc = min(mj_data.ncon, self.max_contacts)
        force = np.zeros(6)
        for i in range(nc):
            cp = mj_data.contact[i]
            spos = cp.pos
            mujoco.mj_contactForce(mj_model, mj_data, i, force)
            fn_vec = -cp.frame.reshape(3, 3) @ force[:3]
            fn = np.linalg.norm(fn_vec)
            if fn < 1e-10:
                continue
            direction = fn_vec / fn
            rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, direction)
            strength = np.tanh(fn * self.gain)
            r = strength
            g = strength * (1 - strength)
            b = 1 - strength
            rgba = (r, g, b, ouc.ALPHA.SOLID)
            a = self._arrows[i]
            a.set_rotmat_pos(rotmat, pos=spos)
            a.rgba = rgba
