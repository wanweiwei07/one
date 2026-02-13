import numpy as np
import one.utils.math as oum
import one.robots.base.mech_base as orbmb


class ManipulatorBase(orbmb.MechBase):

    def __init__(self, rotmat=None, pos=None, is_free=False):
        compiled = self.structure._compiled
        if len(compiled.tip_lnks) != 1:
            raise ValueError("ManipulatorBase must have a single tip.")
        super().__init__(rotmat=rotmat, pos=pos, is_free=is_free)
        self._loc_flange_tf = np.eye(4, dtype=np.float32)
        self._loc_tcp_tf = np.eye(4, dtype=np.float32)
        self._chain = self.structure.get_chain(compiled.root_lnk, compiled.tip_lnks[0])
        self._solver = self.get_solver(self._chain)

    def engage(self, ee, engage_tf=None, update=True, auto_tcp=True):
        engage_tf = oum.ensure_tf(engage_tf)
        super().mount(
            child=ee,
            plnk=self.runtime_lnks[-1],
            engage_tf=self._loc_flange_tf @ engage_tf,
        )
        if update:
            self._update_mounting(self._mountings[ee])
        if auto_tcp:
            self._loc_tcp_tf[:] = engage_tf @ ee.loc_tcp_tf

    def set_loc_tcp_rotmat_pos(self, rotmat=None, pos=None):
        self._loc_tcp_tf[:3, :3] = oum.ensure_rotmat(rotmat)
        self._loc_tcp_tf[:3, 3] = oum.ensure_pos(pos)

    def reset_tcp(self):
        self._loc_tcp_tf[:] = np.eye(4, dtype=np.float32)

    def ik_tcp(self, tgt_rotmat, tgt_pos, max_solutions=8):
        tgt_tcp_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_lastlnk_tf = tgt_tcp_tf @ np.linalg.inv(
            self._loc_flange_tf @ self._loc_tcp_tf
        )
        ik_results = self._solver.ik(
            root_rotmat=self.rotmat,
            root_pos=self.pos,
            tgt_rotmat=tgt_lastlnk_tf[:3, :3],
            tgt_pos=tgt_lastlnk_tf[:3, 3],
            max_solutions=max_solutions,
        )
        if len(ik_results) == 0:
            return None
        return_list = []
        for qs_active in ik_results:
            return_list.append(self._chain.embed_active_qs(qs_active, self.qs))
        return return_list

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs=None):
        tgt_tcp_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_lastlnk_tf = tgt_tcp_tf @ np.linalg.inv(
            self._loc_flange_tf @ self._loc_tcp_tf
        )
        if ref_qs is None:
            ref_qs = self.qs
        ik_results = self._solver.ik(
            root_rotmat=self.rotmat,
            root_pos=self.pos,
            tgt_rotmat=tgt_lastlnk_tf[:3, :3],
            tgt_pos=tgt_lastlnk_tf[:3, 3],
            max_solutions=1,
            ref_qs=ref_qs,
        )
        if len(ik_results) == 0:
            return None
        return self._chain.embed_active_qs(ik_results[0], self.qs)

    def clone(self):
        new = super().clone()
        # rebuild manipulator-specific stuff
        new._loc_flange_tf = self._loc_flange_tf.copy()
        new._loc_tcp_tf = self._loc_tcp_tf.copy()
        new._chain = new.structure.get_chain(
            self.structure.compiled.root_lnk, self.structure.compiled.tip_lnks[0]
        )
        new._solver = new.get_solver(new._chain)
        return new
    
    @property
    def gl_flange_tf(self):
        return self.runtime_lnks[-1].tf @ self._loc_flange_tf

    @property
    def gl_tcp_tf(self):
        return self.gl_flange_tf @ self._loc_tcp_tf

    def mount(self, *args, **kwargs):
        """turn off mount() to avoid confusion"""
        raise RuntimeError(
            "Manipulator.mount() is disabled. "
            "Use engage(child, engage_tf) instead."
        )
