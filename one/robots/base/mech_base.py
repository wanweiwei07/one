import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.utils.decorator as oud
import one.robots.base.mech_structure as orbms


class Mounting:
    def __init__(self, child, parent_link, engage_tf):
        self.child = child
        self.plnk = parent_link
        self.engage_tf = engage_tf


class MechBase:
    _structure: orbms.MechStruct = None

    @classmethod
    def _build_structure(cls):
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
        self.wd_lnk_tfarr = np.tile(
            np.eye(4, dtype=np.float32),
            (self._compiled.n_lnks, 1, 1))
        # mountings
        self._mountings = {}
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
        if qs is not None:
            if len(qs) == len(self.qs):
                self.qs[:] = qs
            else:
                raise ValueError(f"Expected {len(self.qs)} qs, got {qs}")
        q_resolved = self._compiled.resolve_all_qs(self.qs)  # TODO: should this be active only?
        rlidx = self._compiled.root_lnk_idx
        self.wd_lnk_tfarr[rlidx][:3, :3] = self._rotmat
        self.wd_lnk_tfarr[rlidx][:3, 3] = self._pos
        # traversal
        for lidx in self._compiled.lnk_ids_traversal_order:
            if lidx == rlidx:
                continue
            plidx = self._compiled.plidx_of_lidx[lidx]
            pjidx = self._compiled.pjidx_of_lidx[lidx]
            jnt = self.structure.jnts[pjidx]
            plnk_tfmat = self.wd_lnk_tfarr[plidx]
            jtfq = (self._compiled.jtf0_by_idx[pjidx] @
                         jnt.motion_tf(q_resolved[pjidx]))
            self.wd_lnk_tfarr[lidx] = plnk_tfmat @ jtfq
        self._update_runtime()
        return self.wd_lnk_tfarr

    def mount(self, child, plnk, engage_tf=None):
        # TODO updated attach_to?
        if child is self:
            raise ValueError("Self-mounting not allowed")
        if child in self._mountings:
            raise ValueError("Child already mounted")
        if not child.is_free:
            raise ValueError("Child is not free")
        if engage_tf is None:
            engage_tf = np.eye(4, dtype=np.float32)
        else:
            engage_tf = np.asarray(engage_tf, dtype=np.float32)
        self._mountings[child] = Mounting(child, plnk, engage_tf)
        child.is_free = False

    def unmount(self, child):
        try:
            m = self._mountings.pop(child)
        except KeyError:
            raise ValueError("Child not mounted")
        child.is_free = True
        return m

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
        new.wd_lnk_tfarr = self.wd_lnk_tfarr.copy()
        new._mountings = {}
        for k, m in self._mountings.items():
            child = m.child.clone()
            plidx = self.runtime_lidx_map[m.plnk]
            plink = new.runtime_lnks[plidx]
            new._mountings[child] = Mounting(
                child, plink, m.engage_tf.copy())
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
            lnk.tf = self.wd_lnk_tfarr[i]
        # update mountings
        for m in self._mountings.values():
            self._update_mounting(m)

    def _update_mounting(self, mounting):
        child_tf = mounting.plnk.tf @ mounting.engage_tf
        if isinstance(mounting.child, MechBase):
            mounting.child._rotmat[:] = child_tf[:3, :3]
            mounting.child._pos[:] = child_tf[:3, 3]
            mounting.child.fk()
        else:
            mounting.child.tf = child_tf
