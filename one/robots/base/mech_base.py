import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.utils.decorator as oud
import one.robots.base.mech_structure as orbms


class Mounting:
    def __init__(self, child, parent_link, engage_tfmat):
        self.child = child
        self.plnk = parent_link
        self.engage_tfmat = engage_tfmat


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

    def __init__(self, base_rotmat=None, base_pos=None, qs=None):
        self._compiled = self.structure._compiled
        self.base_rotmat = oum.ensure_rotmat(base_rotmat)
        self.base_pos = oum.ensure_pos(base_pos)
        if qs is None:
            self.qs = np.zeros(self._compiled.n_jnts, dtype=np.float32)
        else:
            self.qs = np.asarray(qs, dtype=np.float32)
        # runtime geometry
        self.runtime_lnks = [lnk.clone() for lnk in self.structure.lnks]
        # FK cache
        self.lnk_tfmat_arr = np.tile(np.eye(4, dtype=np.float32),
                                     (self._compiled.n_lnks, 1, 1))
        # mountings
        self._mountings: dict[object, Mounting] = {}
        # home
        self.home_qs = self.qs.copy()
        self.fk(update=True)

    def attach_to(self, scene):
        scene.add(self)
        for m in self._mountings.values():
            scene.add(m.child)

    def detach_from(self, scene):
        for m in self._mountings.values():
            scene.remove(m.child)
        scene.remove(self)

    def set_base_rotmat_pos(self, rotmat, pos):
        self.base_rotmat[:] = oum.ensure_rotmat(rotmat)
        self.base_pos[:] = oum.ensure_pos(pos)
        self.fk(update=True)

    def fk(self, qs=None, update=True):
        if qs is not None and len(qs) == len(self.qs):
            self.qs[:] = qs
        q_resolved = self._compiled.resolve_all_qs(self.qs)  # TODO: should this be active only?
        rlidx = self._compiled.root_lnk_idx
        self.lnk_tfmat_arr[rlidx][:3, :3] = self.base_rotmat
        self.lnk_tfmat_arr[rlidx][:3, 3] = self.base_pos
        # traversal
        for lidx in self._compiled.lnk_ids_traversal_order:
            if lidx == rlidx:
                continue
            plidx = self._compiled.plidx_of_lidx[lidx]
            pjidx = self._compiled.pjidx_of_lidx[lidx]
            jnt = self.structure.jnts[pjidx]
            plnk_tfmat = self.lnk_tfmat_arr[plidx]
            loc_tfmat = (self._compiled.jotfmat_by_idx[pjidx] @
                         jnt.motion_tfmat(q_resolved[pjidx]))
            self.lnk_tfmat_arr[lidx] = plnk_tfmat @ loc_tfmat
        if update:
            self._update_runtime()
        return self.lnk_tfmat_arr

    def mount(self, child, plnk, engage_tfmat=None):
        if child in self._mountings or child is self:
            raise ValueError("Child already mounted or self-mounting")
        if engage_tfmat is None:
            engage_tfmat = np.eye(4, dtype=np.float32)
        else:
            engage_tfmat = np.asarray(engage_tfmat, dtype=np.float32)
        self._mountings[child] = Mounting(child, plnk, engage_tfmat)

    def unmount(self, child):
        try:
            return self._mountings.pop(child)
        except KeyError:
            raise ValueError("Child not mounted")

    def get_lnk_tfmat(self, lnk):
        lidx = self._compiled.lidx_map[lnk]
        return self.lnk_tfmat_arr[lidx]

    def clone(self):
        """DOES NOT clone the affiliated scene"""
        new = self.__class__.__new__(self.__class__)
        new._compiled = self._compiled
        new.base_rotmat = self.base_rotmat.copy()
        new.base_pos = self.base_pos.copy()
        new.qs = self.qs.copy()
        new.runtime_lnks = [lnk.clone() for lnk in self.runtime_lnks]
        new.lnk_tfmat_arr = self.lnk_tfmat_arr.copy()
        new._mountings = {}
        for k, m in self._mountings.items():
            child = m.child.clone()
            plidx = self._compiled.lidx_map[m.plnk]
            new_plnk = new.runtime_lnks[plidx]
            new._mountings[child] = Mounting(
                child, new_plnk, m.engage_tfmat.copy())
        new.home_qs = self.home_qs.copy()
        return new

    def _update_mounting(self, mounting: Mounting):
        parent_tf = self.get_lnk_tfmat(mounting.plnk)
        child_tf = parent_tf @ mounting.engage_tfmat
        if isinstance(mounting.child, MechBase):
            mounting.child.base_rotmat[:] = child_tf[:3, :3]
            mounting.child.base_pos[:] = child_tf[:3, 3]
            mounting.child.fk(update=True)
        else:
            mounting.child.tfmat = child_tf

    @property
    def base_tfmat(self):
        return oum.tfmat_from_rotmat_pos(self.base_rotmat, self.base_pos)

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

    @property
    def rgb(self):
        return [lnk.rgb for lnk in self.runtime_lnks]

    @rgb.setter
    def rgb(self, value):  # only allow uniform color change
        for lnk in self.runtime_lnks:
            lnk.rgb = value

    @property
    def alpha(self):
        return [lnk.alpha for lnk in self.runtime_lnks]

    @alpha.setter
    def alpha(self, value):  # only allow uniform color change
        for lnk in self.runtime_lnks:
            lnk.alpha = value

    # internal helpers
    def _update_runtime(self):
        # push FK result to runtime links
        for i, lnk in enumerate(self.runtime_lnks):
            lnk.tfmat = self.lnk_tfmat_arr[i]
        # update mountings
        for m in self._mountings.values():
            self._update_mounting(m)
