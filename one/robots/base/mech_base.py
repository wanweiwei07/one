import os
import numpy as np
import one.utils.math as oum
import one.robots.base.mech_structure as orbms
import one.robots.base.tcp as orbt
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
        # named chains (which joints move) + tcps (what point to position)
        self._chains = {}
        self._tcps = {}
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
                # use the registered chain name as a readable hint, falling
                # back to link indices for unnamed (ad-hoc) chains.
                label = getattr(chain, "name", None) or \
                    f"{chain.base_lidx}_{chain.tip_lidx}"
                _data_dir = os.path.join(
                    self.structure.res_dir,
                    "data",
                    f"chain_{label}",
                )
            self._solvers[chain] = orbkis.SELIKSolver(
                chain, _data_dir)
        return self._solvers[chain]

    def get_solver(self, chain):
        if chain not in self._solvers:
            raise ValueError("Solver is not initialized for this chain")
        return self._solvers[chain]

    # ---- chain registry ------------------------------------------------
    def add_chain(self, name, base_lnk, tip_lnk):
        """Register a named chain (which joints move). base/tip are STRUCTURE
        links. The chain object is structure-level + shared (cached by
        get_chain), so it is clone-safe with no remapping."""
        if name in self._chains:
            raise ValueError(f"Chain already defined: {name}")
        c = self.structure.get_chain(base_lnk, tip_lnk)
        c.name = name   # readable hint for solver data dirs / debugging
        self._chains[name] = c
        return c

    def chain(self, name):
        return self._chains[name]

    @property
    def chains(self):
        return dict(self._chains)

    # ---- tcp registry --------------------------------------------------
    def add_tcp(self, name, parent_lnk, loc_tf=None):
        """Define a named tcp (link + local offset) used as an ik target.

        parent_lnk must be a runtime link of this mechanism. A single link
        may host any number of tcps.
        """
        if name in self._tcps:
            raise ValueError(f"TCP already defined: {name}")
        if parent_lnk not in self.runtime_lidx_map:
            raise ValueError("parent_lnk must be a runtime link of this mechanism")
        tcp = orbt.TCP(parent_lnk, loc_tf, name)
        self._tcps[name] = tcp
        return tcp

    def tcp(self, name):
        return self._tcps[name]

    @property
    def tcps(self):
        return dict(self._tcps)

    def toggle_tcp(self, name, color_mat=None, **kwargs):
        """Toggle a coordinate frame for the named tcp: first call shows it
        (attached to the tcp's parent link, so it follows the robot), second
        call removes it. Returns the frame sobj when shown, else None."""
        import one.utils.constant as ouc
        import one.scene.scene_object_primitive as ossop
        tcp = self.tcp(name)
        objs = getattr(self, '_tcp_frame_objs', None)
        if objs is None:
            objs = {}
            self._tcp_frame_objs = objs
        if name in objs:
            objs.pop(name).detach_from(tcp.parent_lnk)
            return None
        if color_mat is None:
            color_mat = ouc.CoordColor.MYC
        f = ossop.frame_from_tf(tcp.loc_tf, color_mat=color_mat, **kwargs)
        f.attach_to(tcp.parent_lnk)
        objs[name] = f
        return f

    # ---- ik verb -------------------------------------------------------
    def _resolve_ik_target(self, chain, tcp):
        """Resolve str names, validate control, and return
        (chain, tcp, base_tf, t_tip2tcp) shared by ik / ik_partial.

        chain  : name (str, resolved against this mechanism's chain registry)
                 or a KinematicChain object.
        tcp    : name (str, this mechanism's tcp registry) or a TCP object
                 (object form for a foreign/child registry, e.g. an engaged
                 gripper's tcp, or a transient target).
        The chain-tip -> tcp offset is read from current FK (a rigid constant
        for this solve), so tcp need not sit on chain.tip.
        """
        if isinstance(chain, str):
            chain = self.chain(chain)
        if isinstance(tcp, str):
            tcp = self.tcp(tcp)
        if not self._chain_controls_tcp(chain, tcp):
            raise ValueError(
                "This chain does not control the tcp: tcp.parent_lnk is "
                "neither downstream of the chain tip in this mechanism, nor "
                "on a child mounted downstream of the chain tip.")
        base_tf = self.runtime_lnks[chain.base_lidx].tf
        tip_tf = self.runtime_lnks[chain.tip_lidx].tf
        t_tip2tcp = np.linalg.inv(tip_tf) @ tcp.tf
        return chain, tcp, base_tf, t_tip2tcp

    def ik(self, chain, tcp, tgt_rotmat, tgt_pos,
           max_solutions=8, ref_qs=None, **kwargs):
        """Full-pose IK: solve so ``tcp`` reaches the 6-DOF target pose, using
        ``chain``. Returns a list of full qs vectors (empty if unreachable).
        For under-constrained targets (position-only / axis-direction) use
        ``ik_partial``."""
        chain, tcp, base_tf, t_tip2tcp = self._resolve_ik_target(chain, tcp)
        tgt_tcp_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_tip_tf = tgt_tcp_tf @ np.linalg.inv(t_tip2tcp)
        solver = self._init_solver(chain)
        results = solver.ik(
            root_rotmat=base_tf[:3, :3],
            root_pos=base_tf[:3, 3],
            tgt_rotmat=tgt_tip_tf[:3, :3],
            tgt_pos=tgt_tip_tf[:3, 3],
            max_solutions=max_solutions,
            ref_qs=ref_qs,
            **kwargs,
        )
        return [chain.embed_active_qs(q, self.qs) for q in results]

    def ik_partial(self, chain, tcp, tgt_pos=None, axis_constraints=None,
                   max_solutions=1, ref_qs=None, seed_count=8,
                   pos_weight=1.0, axis_weight=0.2,
                   pos_tol=1e-4, axis_tol=1e-3, max_iter=200,
                   return_infos=False, **kwargs):
        """Partial IK: under-constrained target -- a position and/or
        axis-direction constraints, orientation otherwise free (e.g. point a
        camera's z axis at a target, roll free). Always numerical; the chain's
        solver must support ik_partial (the analytic main-chain solver does
        not).

        Returns a list of full qs vectors (empty if unreachable); with
        ``return_infos=True`` returns ``(qs_list, infos)``.
        """
        if tgt_pos is None and axis_constraints is None:
            raise ValueError('tgt_pos or axis_constraints must be provided')
        chain, tcp, base_tf, t_tip2tcp = self._resolve_ik_target(chain, tcp)
        solver = self._init_solver(chain)
        if not hasattr(solver, 'ik_partial'):
            raise TypeError(
                f"Solver {type(solver).__name__} does not support partial IK; "
                "use a numerical chain (not the analytic main chain).")
        ac = oum.parse_axis_constraints(axis_constraints)
        tgt_pos = None if tgt_pos is None else np.asarray(tgt_pos, dtype=np.float32)
        if ref_qs is None:
            ref_qs = chain.extract_active_qs(self.qs)
        tgt_rotmat_hint = oum.rotmat_from_axis_constraints(ac, ref_rotmat=tcp.rotmat)
        results, infos = solver.ik_partial(
            root_rotmat=base_tf[:3, :3],
            root_pos=base_tf[:3, 3],
            tgt_pos=tgt_pos,
            axis_constraints=ac,
            loc_tf=t_tip2tcp,
            tgt_rotmat_hint=tgt_rotmat_hint,
            max_solutions=max_solutions,
            ref_qs=ref_qs,
            max_iter=max_iter,
            seed_count=seed_count,
            return_infos=True,
            pos_weight=pos_weight,
            axis_weight=axis_weight,
            tol_pos=pos_tol,
            tol_axis=axis_tol,
            **kwargs,
        )
        qs_list = [chain.embed_active_qs(q, self.qs) for q in results]
        return (qs_list, infos) if return_infos else qs_list

    def _chain_controls_tcp(self, chain, tcp):
        """True if moving ``chain`` actually moves ``tcp``.

        Two cases:
        - tcp on one of this mechanism's runtime links: the link must be
          chain.tip or a descendant of it.
        - tcp on a link of a child mounted on this mechanism (e.g. a gripper
          engaged on the flange): the mount's parent link must be chain.tip
          or a descendant of it (the child rides rigidly on it).
        """
        lnk = tcp.parent_lnk
        if lnk in self.runtime_lidx_map:
            return self._lidx_downstream_of_tip(self.runtime_lidx_map[lnk],
                                                chain.tip_lidx)
        for m in self._mountings.values():
            child = m.child
            child_map = getattr(child, 'runtime_lidx_map', None)
            if child_map is not None and lnk in child_map:
                return self._lidx_downstream_of_tip(
                    self.runtime_lidx_map[m.plnk], chain.tip_lidx)
        return False

    def _lidx_downstream_of_tip(self, lidx, tip_lidx):
        """True if link lidx == tip_lidx or is a descendant of it."""
        compiled = self._compiled
        cur = lidx
        while cur >= 0:
            if cur == tip_lidx:
                return True
            cur = compiled.plidx_of_lidx[cur]
        return False

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
        # chains are structure-level + shared -> copy refs, no remap
        new._chains = dict(self._chains)
        new._tcps = {}
        for name, tcp in self._tcps.items():
            plidx = self.runtime_lidx_map[tcp.parent_lnk]
            new._tcps[name] = tcp.copy(parent_lnk=new.runtime_lnks[plidx])
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
