import os
import numpy as np
import one.utils.math as oum
import one.robots.base.mech_structure as orbms
import one.robots.base.kine.numik_sel as orbkis


class Mounting:
    def __init__(self, child, parent_link, loc_tf):
        self.child = child
        self.plnk = parent_link
        self.loc_tf = loc_tf


class MechBase:
    _structure: orbms.MechStruct = None

    @classmethod
    def _build_structure(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def structure(self):
        cls = type(self)
        if cls._structure is None:
            cls._structure = cls._build_structure()
        return cls._structure

    def __init__(self, rotmat=None, pos=None,
                 home_qs=None, is_free=True):
        self._compiled = self.structure._compiled
        self._rotmat = oum.ensure_rotmat(rotmat)
        self._pos = oum.ensure_pos(pos)
        if home_qs is None:
            self.qs = np.zeros(
                self._compiled.n_jnts, dtype=np.float32)
        else:
            self.qs = np.asarray(home_qs, dtype=np.float32)
        # runtime geom
        self.runtime_lnks = [
            lnk.clone() for lnk in self.structure.lnks]
        self.runtime_lidx_map = {
            lnk: i for i, lnk in enumerate(self.runtime_lnks)}
        # FK cache
        self.gl_lnk_tfarr = np.tile(
            np.eye(4, dtype=np.float32),
            (self._compiled.n_lnks, 1, 1))
        # mountings
        self._mountings = {}
        # solvers
        self._solvers = {}
        # home
        self._home_qs = self.qs.copy()
        # free root lnk
        ridx = self.structure.compiled.root_lnk_idx
        self.runtime_lnks[ridx]._is_free = is_free
        self.fk()

    def attach_to(self, scene):
        scene.add(self)
        for m in self._mountings.values():
            m.child.attach_to(scene)

    def detach_from(self, scene):
        for m in self._mountings.values():
            scene.remove(m.child)
        scene.remove(self)

    def set_rotmat_pos(self, rotmat=None, pos=None):
        self._rotmat[:] = oum.ensure_rotmat(rotmat)
        self._pos[:] = oum.ensure_pos(pos)
        self.fk()

    def fk(self, qs=None):
        compiled = self._compiled
        if qs is not None:
            n_qs = len(qs)
            if n_qs == len(self.qs):
                self.qs[:] = qs
            elif n_qs == compiled.n_active_jnts:
                self.qs[compiled.active_jnt_ids_mask] = qs
            else:
                raise ValueError(
                    f'Expected {len(self.qs)} full qs or '
                    f'{compiled.n_active_jnts} active qs, got {n_qs}'
                )
        q_resolved = compiled.resolve_all_qs(self.qs)
        rlidx = compiled.root_lnk_idx
        self.gl_lnk_tfarr[rlidx][:3, :3] = self._rotmat
        self.gl_lnk_tfarr[rlidx][:3, 3] = self._pos
        # traversal
        for lidx in compiled.lnk_ids_traversal_order:
            if lidx == rlidx:
                continue
            plidx = compiled.plidx_of_lidx[lidx]
            pjidx = compiled.pjidx_of_lidx[lidx]
            jnt = self.structure.jnts[pjidx]
            plnk_tfmat = self.gl_lnk_tfarr[plidx]
            jtfq = (compiled.jtf0_by_idx[pjidx] @
                    jnt.motion_tf(q_resolved[pjidx]))
            self.gl_lnk_tfarr[lidx] = plnk_tfmat @ jtfq
        self._update_runtime()
        return self.gl_lnk_tfarr

    def mount(self, child, plnk, loc_tf=None, update=False):
        """
            Note: child is not attached to the scene when this is called
            Caller is responsible for attaching the child to a scene
        """
        if child is self:
            raise ValueError("Self-mounting not allowed")
        if child in self._mountings:
            raise ValueError("Child already mounted")
        if not child.is_free:
            raise ValueError("Child is not free")
        if loc_tf is None:
            loc_tf = np.eye(4, dtype=np.float32)
        else:
            loc_tf = np.asarray(loc_tf, dtype=np.float32)
        self._mountings[child] = Mounting(child, plnk, loc_tf)
        child.is_free = False
        if update:
            self._update_mounting(self._mountings[child])

    def unmount(self, child):
        try:
            m = self._mountings.pop(child)
        except KeyError:
            raise ValueError("Child not mounted")
        child.is_free = True
        return m

    def _init_solver(self, chain):
        if chain not in self._solvers:
            if (chain.base_lidx == self.structure.compiled.root_lnk_idx and
                    chain.tip_lidx in self.structure.compiled.tip_lnk_ids):
                _data_dir = os.path.join(self.structure.res_dir, "data")
            else:
                _data_dir = os.path.join(
                    self.structure.res_dir,
                    "data",
                    f"chain_{chain.base_lidx}_{chain.tip_lidx}",
                )
            self._solvers[chain] = orbkis.SELIKSolver(
                chain, _data_dir)
        return self._solvers[chain]

    def get_solver(self, chain):
        if chain not in self._solvers:
            raise ValueError("Solver is not initialized for this chain")
        return self._solvers[chain]

    def clone(self):
        """DOES NOT clone the affiliated scene"""
        new = self.__class__.__new__(self.__class__)
        new._compiled = self._compiled
        new._rotmat = self._rotmat.copy()
        new._pos = self._pos.copy()
        new.qs = self.qs.copy()
        new.runtime_lnks = [lnk.clone() for lnk in self.runtime_lnks]
        new.runtime_lidx_map = {
            lnk: i for i, lnk in enumerate(new.runtime_lnks)}
        new.gl_lnk_tfarr = self.gl_lnk_tfarr.copy()
        new._mountings = {}
        new._solvers = self._solvers  # solvers can be shared
        for k, m in self._mountings.items():
            child = m.child.clone()
            plidx = self.runtime_lidx_map[m.plnk]
            plink = new.runtime_lnks[plidx]
            new._mountings[child] = Mounting(
                child, plink, m.loc_tf.copy())
        new._home_qs = self._home_qs.copy()
        return new

    @property
    def ndof(self):
        return self._compiled.n_active_jnts

    @property
    def runtime_root_lnk(self):
        ridx = self.structure.compiled.root_lnk_idx
        return self.runtime_lnks[ridx]

    @property
    def is_free(self):
        return self.runtime_root_lnk.is_free

    @is_free.setter
    def is_free(self, flag):
        self.runtime_root_lnk.is_free = flag

    @property
    def home_qs(self):
        return self._home_qs.copy()

    @home_qs.setter
    def home_qs(self, value):
        self._home_qs[:] = np.asarray(value, dtype=np.float32)

    @property
    def tf(self):
        return oum.tf_from_rotmat_pos(self._rotmat, self._pos)

    @property
    def rotmat(self):
        return self._rotmat.copy()

    @rotmat.setter
    def rotmat(self, value):  # TODO: delay update
        self._rotmat[:] = oum.ensure_rotmat(value)
        self.fk()

    @property
    def quat(self):
        return oum.quat_from_rotmat(self._rotmat)

    @property
    def pos(self):
        return self._pos.copy()

    @pos.setter
    def pos(self, value):  # TODO: delay update
        self._pos[:] = oum.ensure_pos(value)
        self.fk()

    @property
    def toggle_render_collision(self):
        return self.runtime_lnks[0].toggle_render_collision

    @toggle_render_collision.setter
    def toggle_render_collision(self, flag=True):
        for lnk in self.runtime_lnks:
            lnk.toggle_render_collision = flag

    @property
    def rgba(self):
        return [lnk.rgba for lnk in self.runtime_lnks]

    @rgba.setter
    def rgba(self, value):  # only allow uniform color change
        for lnk in self.runtime_lnks:
            lnk.rgba = value
        for m in self._mountings.values():
            m.child.rgba = value

    @property
    def rgb(self):
        return [lnk.rgb for lnk in self.runtime_lnks]

    @rgb.setter
    def rgb(self, value):  # only allow uniform color change
        for lnk in self.runtime_lnks:
            lnk.rgb = value
        for m in self._mountings.values():
            m.child.rgb = value

    @property
    def alpha(self):
        return [lnk.alpha for lnk in self.runtime_lnks]

    @alpha.setter
    def alpha(self, value):  # only allow uniform color change
        for lnk in self.runtime_lnks:
            lnk.alpha = value
        for m in self._mountings.values():
            m.child.alpha = value

    # internal helpers
    def _update_runtime(self):
        # push FK result to runtime links
        for i, lnk in enumerate(self.runtime_lnks):
            lnk.tf = self.gl_lnk_tfarr[i]
        # update mountings
        for m in self._mountings.values():
            self._update_mounting(m)

    def _update_mounting(self, mounting):
        child_tf = mounting.plnk.tf @ mounting.loc_tf
        if isinstance(mounting.child, MechBase):
            mounting.child._rotmat[:] = child_tf[:3, :3]
            mounting.child._pos[:] = child_tf[:3, 3]
            mounting.child.fk()
        else:
            mounting.child.tf = child_tf
