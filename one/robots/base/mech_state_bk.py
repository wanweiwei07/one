import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc


class MechState:

    def __init__(self, structure, base_rotmat=None, base_pos=None, qs=None):
        self._compiled = structure._compiled
        self.base_rotmat = oum.ensure_rotmat(base_rotmat)
        self.base_pos = oum.ensure_pos(base_pos)
        if qs is None:
            self.qs = np.zeros(self._compiled.n_jnts, dtype=np.float32)
        else:
            self.qs = np.asarray(qs, dtype=np.float32)
        # TODO: dirty tfmats and auto fk after set qs
        # lnk_ref_tfmat: world transforms of kinematic reference frames after joint motion
        # (indexed by link; equivalent to joint after-motion frames)
        self.lnk_ref_tfmat_arr = np.tile(np.eye(4, dtype=np.float32),
                                         (self._compiled.n_lnks, 1, 1))
        self.runtime_lnks = []
        for lnk in structure.lnks:
            self.runtime_lnks.append(lnk.clone())

    def get_lnk_ref_tfmat(self, lnk):
        idx = self._compiled.lidx_map[lnk]
        return self.lnk_ref_tfmat_arr[idx]

    def fk(self, qs=None):
        if qs is not None:
            self._set_qs(qs)
        q_resolved = self._compiled.resolve_all_qs(self.qs)
        rlnk_idx = self._compiled.root_lnk_idx
        self.lnk_ref_tfmat_arr[rlnk_idx][:3, :3] = self.base_rotmat
        self.lnk_ref_tfmat_arr[rlnk_idx][:3, 3] = self.base_pos
        for lnk_idx in self._compiled.lnk_ids_traversal_order:
            # lnk_idx = self.structure.link_dfs_index_map[lidx]
            if lnk_idx == rlnk_idx:
                continue
            plnk_idx = self._compiled.plidx_of_lidx[lnk_idx]
            pjnt_idx = self._compiled.pjidx_of_lidx[lnk_idx]
            plnk_tfmat = self.lnk_ref_tfmat_arr[plnk_idx]
            loc_tfmat = (self._compiled.jotfmat_by_idx[pjnt_idx] @
                         self._jnt_motion_tfmat(pjnt_idx, q_resolved[pjnt_idx]))
            self.lnk_ref_tfmat_arr[lnk_idx] = plnk_tfmat @ loc_tfmat
        return self.lnk_ref_tfmat_arr

    def update(self):
        for i, lnk in enumerate(self.runtime_lnks):
            lnk.tf = self.lnk_ref_tfmat_arr[i]

    def attach_to(self, scene):
        scene.add(self)

    def remove_from(self, scene):
        for lnk in self.runtime_lnks:
            scene.remove(lnk)

    def clone(self):
        new = MechState.__new__(MechState)  # bypass __init__
        new._compiled = self._compiled
        new.base_rotmat = self.base_rotmat.copy()
        new.base_pos = self.base_pos.copy()
        new.qs = self.qs.copy()
        new.lnk_ref_tfmat_arr = self.lnk_ref_tfmat_arr.copy()
        new.runtime_lnks = [lnk.clone() for lnk in self._compiled._meta.lnks]
        return new

    @property
    def base_tfmat(self):
        return oum.tf_from_rotmat_pos(
            self.base_rotmat, self.base_pos)

    @property
    def toggle_render_collision(self):
        return self.runtime_lnks[0].toggle_render_collision

    @toggle_render_collision.setter
    def toggle_render_collision(self, flag=True):
        for link in self.runtime_lnks:
            link.toggle_render_collision = flag

    @property
    def n_jnts(self):
        return self._compiled.n_jnts

    @property
    def rgba(self):
        return [lnk.rgba for lnk in self.runtime_lnks]

    @rgba.setter
    def rgba(self, value):
        for lnk in self.runtime_lnks:
            lnk.rgba = value

    @property
    def rgb(self):
        return [lnk.rgb for lnk in self.runtime_lnks]

    @rgb.setter
    def rgb(self, value):
        for lnk in self.runtime_lnks:
            lnk.rgb = value

    @property
    def alpha(self):
        return [lnk.alpha for lnk in self.runtime_lnks]

    @alpha.setter
    def alpha(self, value):
        for lnk in self.runtime_lnks:
            lnk.alpha = value

    def _set_qs(self, values):
        # TODO active qs only (joints with no mimic)
        values = np.asarray(values, dtype=np.float32)
        assert len(values) == len(self.qs), f"Expected {len(self.qs)} joint values, got {len(values)}"
        self.qs[:] = values

    def _jnt_motion_tfmat(self, jnt_idx, q):
        jnt_type = self._compiled.jtypes_by_idx[jnt_idx]
        jnt_ax = self._compiled.jax_by_idx[jnt_idx]
        if jnt_type == ouc.JntType.FIXED:
            return np.eye(4, dtype=np.float32)
        if jnt_type == ouc.JntType.REVOLUTE:
            return oum.tf_from_rotmat_pos(rotmat=oum.rotmat_from_axangle(jnt_ax, q))
        if jnt_type == ouc.JntType.PRISMATIC:
            return oum.tf_from_rotmat_pos(pos=jnt_ax * q)
        raise TypeError(f"Unknown joint type: {jnt_type}")
