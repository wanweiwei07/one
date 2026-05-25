import numpy as np

import one.utils.math as oum


class AttachedFrame:
    """A local coordinate frame attached to a runtime link."""

    def __init__(self, parent_lnk, loc_tf=None, chain=None, solver=None):
        self.parent_lnk = parent_lnk
        self.loc_tf = oum.ensure_tf(loc_tf)
        self.chain = chain
        self.solver = solver

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

    def copy(self):
        return AttachedFrame(
            self.parent_lnk, self.loc_tf.copy(), self.chain, self.solver)
