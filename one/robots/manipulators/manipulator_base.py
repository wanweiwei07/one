import numpy as np
import one.utils.math as oum
import one.robots.base.mech_base as orbmb


class ManipulatorBase(orbmb.MechBase):

    def __init__(self, rotmat=None, pos=None, is_free=False, data_dir="data"):
        compiled = self.structure._compiled
        if len(compiled.tip_lnks) != 1:
            raise ValueError("ManipulatorBase must have a single tip.")
        super().__init__(rotmat=rotmat, pos=pos, is_free=is_free)
        self._tcp_tf = np.eye(4, dtype=np.float32)
        self._chain = self.structure.get_chain(compiled.root_lnk,
                                               compiled.tip_lnks[0])
        self._solver = self.structure.get_solver(compiled.root_lnk,
                                                 compiled.tip_lnks[0],
                                                 data_dir)

    def engage(self, ee, engage_tfmat=None,
               update=True, auto_tcp=True):
        super().mount(child=ee,
                      plnk=self.runtime_lnks[-1],
                      engage_tf=engage_tfmat)
        if update:
            self._update_mounting(self._mountings[ee])
        if auto_tcp:
            flange_tfmat = self._mountings[ee].engage_tf
            self._tcp_tf[:] = flange_tfmat @ ee.tcp_tf

    def set_tcp_rotmat_pos(self, rotmat=None, pos=None):
        self._tcp_tf[:3, :3] = oum.ensure_tf(rotmat)
        self._tcp_tf[:3, 3] = oum.ensure_pos(pos)

    def reset_tcp(self):
        self._tcp_tf[:] = np.eye(4, dtype=np.float32)

    def ik_tcp(self, tgt_rotmat, tgt_pos, max_solutions=8):
        tgt_tcp_tfmat = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_flange_tfmat = tgt_tcp_tfmat @ np.linalg.inv(self._tcp_tf)
        result_list = self._solver.ik(
            root_rotmat=self.rotmat,
            root_pos=self.pos,
            tgt_rotmat=tgt_flange_tfmat[:3, :3],
            tgt_pos=tgt_flange_tfmat[:3, 3],
            max_solutions=max_solutions)
        if len(result_list) == 0:
            return result_list
        return_list = []
        for result in result_list:
            return_list.append(self._chain.embed_active_qs(result[0], self.qs))
        return return_list

    def clone(self):
        new = super().clone()
        # rebuild manipulator-specific stuff
        new._tcp_tf = self._tcp_tf.copy()
        new._chain = new.structure.get_chain(
            self.structure.compiled.root_lnk,
            self.structure.compiled.tip_lnks[0])
        new._solver = new.structure.get_solver(
            self.structure.compiled.root_lnk,
            self.structure.compiled.tip_lnks[0],
            self._solver._data_dir)
        return new

    @property
    def wd_tcp_tf(self):
        return self.runtime_lnks[-1].tf @ self._tcp_tf

    def mount(self, *args, **kwargs):
        """turn off mount() to avoid confusion"""
        raise RuntimeError("Manipulator.mount() is disabled. "
                           "Use engage(child, engage_tfmat) instead.")
