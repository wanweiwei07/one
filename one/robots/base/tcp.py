import numpy as np

import one.utils.math as oum


class TCP:
    """A local frame (link + offset) used as an IK target point.

    A TCP is *what* you position; a KinematicChain is *which* joints move.
    They are decoupled: one runtime link can host many TCPs, and a TCP is not
    bound to any chain. It holds no qs / chain / solver -- the real state lives
    on the owning MechBase.

    ``name`` is optional metadata (for registry lookup / visualization);
    pass the TCP object itself to ``MechBase.ik`` -- no string needed there.
    """

    def __init__(self, parent_lnk, loc_tf=None, name=None):
        self.parent_lnk = parent_lnk
        self.loc_tf = oum.ensure_tf(loc_tf)
        self.name = name

    @property
    def tf(self):
        return self.parent_lnk.tf @ self.loc_tf

    @property
    def pos(self):
        return self.tf[:3, 3].copy()

    @property
    def rotmat(self):
        return self.tf[:3, :3].copy()

    def set_loc_tf(self, loc_tf=None):
        self.loc_tf = oum.ensure_tf(loc_tf)

    def set_loc_rotmat_pos(self, rotmat=None, pos=None):
        self.loc_tf = oum.tf_from_rotmat_pos(rotmat, pos)

    def copy(self, parent_lnk=None):
        return TCP(parent_lnk or self.parent_lnk, self.loc_tf.copy(), self.name)
