import numpy as np

import one.utils.math as oum
from one.robots.base.attached_frame import AttachedFrame
import one.robots.base.mech_base as orbmb


class ManipulatorBase(orbmb.MechBase):

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_free=False):
        compiled = self.structure._compiled
        if len(compiled.tip_lnks) != 1:
            raise ValueError("ManipulatorBase must have a single tip.")
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_free=is_free)
        self._loc_flange_tf = np.eye(4, dtype=np.float32)
        self._loc_tcp_tf = np.eye(4, dtype=np.float32)
        self._main_chain = self.structure.get_chain(compiled.root_lnk,
                                                    compiled.tip_lnks[0])
        self._attached_frames = {}
        self._init_solver(self._main_chain)

    def engage(self, ee, loc_tf=None, update=True, auto_tcp=True):
        loc_tf = oum.ensure_tf(loc_tf)
        super().mount(
            child=ee,
            plnk=self.runtime_lnks[-1],
            loc_tf=self._loc_flange_tf @ loc_tf,
        )
        if update:
            self._update_mounting(self._mountings[ee])
        if auto_tcp:
            self._loc_tcp_tf[:] = loc_tf @ ee.loc_tcp_tf

    def attach_sensor(self, sensor, lnk, loc_tf=None, update=True):
        """Attach a sensor or auxiliary object to a runtime link.

        This does not update the manipulator TCP. Scene membership remains
        explicit: attach the manipulator to a scene after attaching sensors,
        or attach the sensor to the scene yourself for runtime additions.
        """
        loc_tf = oum.ensure_tf(loc_tf)
        super().mount(
            child=sensor,
            plnk=lnk,
            loc_tf=loc_tf,
            update=update,
        )
        return sensor

    def define_attached_frame(self, name, parent_lnk, loc_tf=None):
        """Define a named local frame on a runtime link."""
        if name in self._attached_frames:
            raise ValueError(f'Attached frame already defined: {name}')
        if parent_lnk not in self.runtime_lidx_map:
            raise ValueError('parent_lnk must be a runtime link of this manipulator')
        parent_lidx = self.runtime_lidx_map[parent_lnk]
        parent_struct_lnk = self.structure.lnks[parent_lidx]
        chain = self.structure.get_chain(
            self.structure.compiled.root_lnk, parent_struct_lnk)
        solver = self._init_solver(chain)
        frame = AttachedFrame(parent_lnk, loc_tf, chain=chain, solver=solver)
        self._attached_frames[name] = frame
        return frame

    def get_attached_frame(self, name):
        return self._attached_frames[name]

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
        ik_results = self.get_solver(self._main_chain).ik(
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
            return_list.append(self._main_chain.embed_active_qs(qs_active, self.qs))
        return return_list

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs=None):
        tgt_tcp_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_lastlnk_tf = tgt_tcp_tf @ np.linalg.inv(
            self._loc_flange_tf @ self._loc_tcp_tf
        )
        if ref_qs is None:
            ref_qs = self.qs
        ik_results = self.get_solver(self._main_chain).ik(
            root_rotmat=self.rotmat,
            root_pos=self.pos,
            tgt_rotmat=tgt_lastlnk_tf[:3, :3],
            tgt_pos=tgt_lastlnk_tf[:3, 3],
            max_solutions=1,
            ref_qs=ref_qs,
        )
        if len(ik_results) == 0:
            return None
        return self._main_chain.embed_active_qs(ik_results[0], self.qs)

    def ik_attached_frame(
            self,
            frame,
            target_pos=None,
            axis_constraints=None,
            selik_seed_count=8,
            pos_weight=1.0,
            axis_weight=0.2,
            pos_tol=1e-4,
            axis_tol=1e-3,
            max_nfev=200):
        """Numerically solve IK for an AttachedFrame-like object.

        Args:
            frame: AttachedFrame object.
            target_pos: Optional target frame position, shape (3,).
            axis_constraints: Optional orientation constraints. Supported:
                - [(local_axis, target_axis), ...]
                - {'x': target_axis, 'y': target_axis, 'z': target_axis}
                - a 3x3 target rotmat, equivalent to constraining x/y/z.

        Returns:
            tuple: (full_qs, solver_info, ok)
        """
        if target_pos is None and axis_constraints is None:
            raise ValueError('target_pos or axis_constraints must be provided')

        if not isinstance(frame, AttachedFrame):
            raise ValueError('frame must be an AttachedFrame object')
        target_pos = None if target_pos is None else np.asarray(
            target_pos, dtype=np.float32)
        axis_constraints = oum.parse_axis_constraints(axis_constraints)
        ik_chain = frame.chain
        solver = frame.solver
        if ik_chain is None or solver is None or ik_chain.n_active_jnts == 0:
            return None, None, False
        ref_qs = ik_chain.extract_active_qs(self.qs)
        tgt_rotmat_hint = oum.rotmat_from_axis_constraints(
            axis_constraints, ref_rotmat=frame.rotmat)
        ik_results, infos = solver.ik_partial(
            root_rotmat=self.rotmat,
            root_pos=self.pos,
            tgt_pos=target_pos,
            axis_constraints=axis_constraints,
            loc_tf=frame.loc_tf,
            tgt_rotmat_hint=tgt_rotmat_hint,
            max_solutions=1,
            ref_qs=ref_qs,
            max_iter=max_nfev,
            seed_count=selik_seed_count,
            return_infos=True,
            pos_weight=pos_weight,
            axis_weight=axis_weight,
            tol_pos=pos_tol,
            tol_axis=axis_tol,
        )
        if len(ik_results) == 0:
            info = infos[0] if infos else None
            return None, info, False
        qs = ik_chain.embed_active_qs(ik_results[0], self.qs)
        self.fk(qs=qs)
        return qs, infos[0] if infos else None, True

    def clone(self):
        new = super().clone()
        # rebuild manipulator-specific stuff
        new._loc_flange_tf = self._loc_flange_tf.copy()
        new._loc_tcp_tf = self._loc_tcp_tf.copy()
        new._main_chain = new.structure.get_chain(
            self.structure.compiled.root_lnk,
            self.structure.compiled.tip_lnks[0]
        )
        new._init_solver(new._main_chain)
        new._attached_frames = {}
        for name, frame in self._attached_frames.items():
            parent_lidx = self.runtime_lidx_map[frame.parent_lnk]
            parent_struct_lnk = new.structure.lnks[parent_lidx]
            chain = new.structure.get_chain(
                new.structure.compiled.root_lnk, parent_struct_lnk)
            solver = new._init_solver(chain)
            new._attached_frames[name] = AttachedFrame(
                new.runtime_lnks[parent_lidx], frame.loc_tf.copy(),
                chain=chain, solver=solver)
        return new

    @property
    def attached_frames(self):
        return dict(self._attached_frames)

    @property
    def gl_flange_tf(self):
        return self.runtime_lnks[-1].tf @ self._loc_flange_tf

    @property
    def gl_tcp_tf(self):
        return self.gl_flange_tf @ self._loc_tcp_tf

    def toggle_tcp(self, color_mat=None, **kwargs):
        """Toggle a TCP coordinate frame attached to the flange link.
        First call shows it (follows the robot automatically); second call
        removes it. Returns the frame sobj when shown, else None."""
        import one.utils.constant as ouc
        import one.scene.scene_object_primitive as ossop
        flange_lnk = self.runtime_lnks[-1]
        if getattr(self, "_tcp_frame", None) is not None:
            self._tcp_frame.detach_from(flange_lnk)
            self._tcp_frame = None
            return None
        if color_mat is None:
            color_mat = ouc.CoordColor.MYC
        loc_tcp_tf = self._loc_flange_tf @ self._loc_tcp_tf
        f = ossop.frame_from_tf(loc_tcp_tf, color_mat=color_mat, **kwargs)
        f.attach_to(flange_lnk)
        self._tcp_frame = f
        return f

    def toggle_attached_frames(self, names=None, color_mat=None, **kwargs):
        """Toggle coordinate frames for named AttachedFrame objects."""
        import one.utils.constant as ouc
        import one.scene.scene_object_primitive as ossop
        frame_objs = getattr(self, '_attached_frame_objs', None)
        if frame_objs is not None:
            for name, sobj in frame_objs.items():
                sobj.detach_from(self._attached_frames[name].parent_lnk)
            self._attached_frame_objs = None
            return None
        if color_mat is None:
            color_mat = ouc.CoordColor.MYC
        if names is None:
            names = tuple(self._attached_frames.keys())
        elif isinstance(names, str):
            names = (names,)
        frame_objs = {}
        for name in names:
            frame = self._attached_frames[name]
            sobj = ossop.frame_from_tf(
                frame.loc_tf, color_mat=color_mat, **kwargs)
            sobj.attach_to(frame.parent_lnk)
            frame_objs[name] = sobj
        self._attached_frame_objs = frame_objs
        return frame_objs

    def mount(self, *args, **kwargs):
        """turn off mount() to avoid confusion"""
        raise RuntimeError(
            "Manipulator.mount() is disabled. " 
            "Use engage(child, loc_tf) for end effectors, "
            "or attach_sensor(sensor, lnk, loc_tf) for auxiliary attachments."
        )
