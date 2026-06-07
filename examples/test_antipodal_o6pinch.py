"""Antipodal grasp planning for the Linkerbot O6 dexterous hand's PINCH.

``antipodal`` is written for a parallel-jaw gripper, but a thumb-index pinch is
exactly a 2-point opposing grasp, so we present the pinch to it through a thin
adapter (``O6Pinch``) that exposes the gripper interface antipodal needs:
``grasp_center`` tcp, ``open_dir``, ``jaw_range``, ``contact_pattern``,
``set_jaw_width`` (-> pinch synergy, with the other 3 fingers tucked away) and
``grip_at``.

Shows the best pinch on a small bunny solid, plus the next few as translucent
ghosts. Keys: N = next grasp as the solid one, R = re-plan.
"""
import os

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.geom.loader as ogl
import one.scene.render_model as osrm
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.robots.end_effectors.linkerbot.o6.o6 as o6
from one.grasp.antipodal import antipodal

BUNNY_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bunny_small.stl')
TUCK = (1.3, 1.0)   # mcp/dip curl for the non-pinching middle/ring/pinky


class O6Pinch(o6.O6Left):
    """O6 left hand adapted to the parallel-jaw interface ``antipodal`` expects,
    driving the thumb-index pinch. ``jaw_width`` is the thumb-index opening; the
    middle/ring/pinky fingers are tucked so only the pinch pair is presented."""

    REF_AMOUNT = 0.7   # representative half-pinch where an object sits between
                       # the pads (full pinch is degenerate: the pads touch)

    def __init__(self):
        super().__init__()
        # grasp_center / open_dir from the PAD contact (where an object is
        # actually pinched), not the fingertips: at a half-pinch the closest
        # points between the thumb and index distal meshes are the two pads.
        pad_t, pad_i = self._pad_contacts(self.REF_AMOUNT)
        b_inv = np.linalg.inv(self.runtime_root_lnk.tf)   # world -> hand base
        ct = b_inv[:3, :3] @ pad_t + b_inv[:3, 3]
        ci = b_inv[:3, :3] @ pad_i + b_inv[:3, 3]
        center = ((ct + ci) * 0.5).astype(np.float32)
        # antipodal hardcodes the tcp name 'grasp_center'
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(pos=center))
        # opposition axis = thumb pad -> index pad, in the grasp_center frame
        # (identity rotation, so a hand-base-frame vector)
        axis = ci - ct
        self.open_dir = (axis / (np.linalg.norm(axis) + oum.eps)).astype(np.float32)
        self.jaw_range = np.array([0.0, 0.045], dtype=np.float32)
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self._jaw_w = self.jaw_range[1]
        self.set_jaw_width(self._jaw_w)

    def _world_vs(self, lnk_name):
        c = self._compiled
        lnk = self.runtime_lnks[c.lidx_map[self.structure.lnk_map[lnk_name]]]
        v = lnk.visuals[0]
        vs = np.asarray(v.geom.vs, dtype=np.float64).reshape(-1, 3)
        m = (lnk.tf @ v._tf).astype(np.float64)
        return vs @ m[:3, :3].T + m[:3, 3]

    def _pad_contacts(self, amount):
        """Closest vertex pair between the thumb and index distal meshes at the
        given pinch amount -> the two pad contact points (world frame)."""
        self.pinch(amount)
        a = self._world_vs('lh_thumb_distal')
        b = self._world_vs('lh_index_distal')
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return a[i], b[j]

    def set_jaw_width(self, w):
        jmin, jmax = self.jaw_range
        amount = float(np.clip((jmax - w) / (jmax - jmin + oum.eps), 0.0, 1.0))
        qs = np.zeros(self.ndof, dtype=np.float32)
        nm = {j.name: i for i, j in enumerate(self.structure.jnts)}
        for jname, closed in self.grasp_synergies['pinch'].items():
            qs[nm[jname]] = closed * amount
        for f in ('middle', 'ring', 'pinky'):
            qs[nm[f'lh_{f}_mcp_pitch']] = TUCK[0]
            qs[nm[f'lh_{f}_dip']] = TUCK[1]
        self.fk(qs=qs)
        self._jaw_w = w

    def grip_at(self, tgt_pos, tgt_rotmat, jaw_width):
        loc = self.tcp('grasp_center').loc_tf
        base = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat) @ np.linalg.inv(loc)
        self.set_pos_rotmat(base[:3, 3], base[:3, :3])
        self.set_jaw_width(jaw_width)

    def clone(self):
        new = super().clone()
        new.open_dir = self.open_dir.copy()
        new.jaw_range = self.jaw_range.copy()
        new.contact_pattern = self.contact_pattern.copy()
        new.set_jaw_width(self._jaw_w)
        return new


def make_bunny():
    bunny = osso.SceneObject(collision_type=ouc.CollisionType.MESH, is_free=True)
    bunny.add_visual(
        osrm.RenderModel(geom=ogl.load_geometry(BUNNY_STL), rgb=(0.85, 0.7, 0.6)),
        auto_make_collision=True)
    return bunny


def main():
    base = ovw.World(cam_pos=(0.28, 0.22, 0.18), cam_lookat_pos=(0.0, 0.0, 0.02))
    ossop.frame(length_scale=0.2).attach_to(base.scene)

    grasper = O6Pinch()
    bunny = make_bunny()
    bunny.attach_to(base.scene)

    state = {'grasps': [], 'cur': 0, 'solid': None, 'ghosts': []}

    def clear():
        if state['solid'] is not None:
            state['solid'].detach_from(base.scene)
            state['solid'] = None
        for gh in state['ghosts']:
            gh.detach_from(base.scene)
        state['ghosts'] = []

    def show(idx):
        clear()
        grasps = state['grasps']
        if not grasps:
            return
        idx %= len(grasps)
        state['cur'] = idx
        # a few lower-ranked candidates as translucent ghosts
        for pose, _pre, jw, _sc in grasps[1:6]:
            gh = grasper.clone()
            gh.grip_at(pose[:3, 3], pose[:3, :3], jw)
            gh.alpha = 0.2
            gh.attach_to(base.scene)
            state['ghosts'].append(gh)
        # the selected grasp solid
        pose, _pre, jw, _sc = grasps[idx]
        solid = grasper.clone()
        solid.grip_at(pose[:3, 3], pose[:3, :3], jw)
        solid.attach_to(base.scene)
        state['solid'] = solid

    def replan():
        grasps = antipodal(grasper, bunny, density=0.0008, normal_tol_deg=25,
                           roll_step_deg=30, max_grasps=40, clearance=0.003)
        state['grasps'] = grasps
        print(f'antipodal pinch grasps: {len(grasps)} '
              f'(N = next, R = re-plan)')
        show(0)

    replan()

    import pyglet.window.key as key

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            show(state['cur'] + 1)
        if base.input_manager.is_key_pressed_edge(key.R):
            replan()

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == '__main__':
    main()
