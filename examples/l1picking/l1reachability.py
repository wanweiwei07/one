"""3D reachability / workspace map of the L1 left arm (6-DOF, waist frozen at
home), targeting the hand's ``pinch_center`` tcp.

Method -- forward-kinematics sampling into a 3D voxel grid (a "capability map").
We sample many random arm configs, FK the tcp (vectorized product-of-exponentials,
~1M/sec), and bin the hits into voxels. A voxel's hit count is its reachability
density -- a proxy for orientation freedom / dexterity (more configs reach it ->
more poses available there). Position reachability needs *some* orientation, so
FK-binning is the right tool; probing each voxel with IK + a guessed orientation
badly under-reports near the workspace boundary.

Viewer:
  - faint full cloud of every reachable voxel (colored red->green by density)
  - one bright axis-aligned SLICE you can step through to inspect cross-sections
Keys:  X / Y / Z      pick the slice axis (slice plane is perpendicular to it)
       , / .  (or [ ] or Up/Down)  step the slice along that axis
       A              toggle the faint full cloud
       O              show everything (no slice)
Headless (prints stats + a mid slice per axis):  ONE_HEADLESS=1
"""
import os
import sys
import builtins

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.robots.humanoids.linx.l1.l1 as l1                  # noqa: E402

X_RANGE = (-0.1, 0.6)
Y_RANGE = (-0.3, 0.9)
Z_RANGE = (0.6, 1.7)
RES = 0.04                # voxel edge (m)
N_SAMPLES = 1_000_000
CHAIN = 'left_arm'
TCP_NAME = 'pinch_center'
_AXES = ('x', 'y', 'z')


def make_poe(robot):
    """Vectorized FK of the CHAIN tip TCP via product-of-exponentials, with the
    waist frozen at home. Returns (poe(q_batch)->(M,3) world positions, chain)."""
    ch = robot.chain(CHAIN)
    tcp = robot.left_hand.tcp(TCP_NAME)
    robot.fk(qs=np.zeros(robot.qs.shape, dtype=np.float32))   # zero config
    root_world = robot.lnk('waist_link2').tf.astype(np.float64).copy()
    tcp0 = (np.linalg.inv(root_world) @ tcp.tf.astype(np.float64)).copy()
    I3 = np.eye(3)

    def skew(w):
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    Ks, oris = [], []
    for i in range(6):
        a = np.asarray(ch.axes[i], dtype=np.float64)
        Ks.append(skew(a / np.linalg.norm(a)))
        oris.append(np.asarray(ch.origins[i], dtype=np.float64))

    def poe(qb):
        M = qb.shape[0]
        T = np.tile(np.eye(4), (M, 1, 1))
        for i in range(6):
            K = Ks[i]
            q = qb[:, i]
            c = np.cos(q)[:, None, None]
            s = np.sin(q)[:, None, None]
            R = I3[None] + s * K[None] + (1.0 - c) * (K @ K)[None]
            p = np.einsum('mij,j->mi', I3[None] - R, oris[i])
            E = np.tile(np.eye(4), (M, 1, 1))
            E[:, :3, :3] = R
            E[:, :3, 3] = p
            T = T @ E
        W = root_world[None] @ (T @ tcp0)
        return W[:, :3, 3]

    return poe, ch


def build_grid(poe, ch, n_samples=N_SAMPLES, seed=0):
    lo = np.asarray(ch.lmt_lo, dtype=np.float64)
    hi = np.asarray(ch.lmt_up, dtype=np.float64)
    rng = np.random.default_rng(seed)
    x0, x1 = X_RANGE
    y0, y1 = Y_RANGE
    z0, z1 = Z_RANGE
    nx = int(np.ceil((x1 - x0) / RES))
    ny = int(np.ceil((y1 - y0) / RES))
    nz = int(np.ceil((z1 - z0) / RES))
    counts = np.zeros((nx, ny, nz), dtype=np.int32)
    batch = 200_000
    for start in range(0, n_samples, batch):
        m = min(batch, n_samples - start)
        P = poe(rng.uniform(lo, hi, size=(m, 6)))
        inb = ((P[:, 0] >= x0) & (P[:, 0] < x1) &
               (P[:, 1] >= y0) & (P[:, 1] < y1) &
               (P[:, 2] >= z0) & (P[:, 2] < z1))
        Q = P[inb]
        ix = ((Q[:, 0] - x0) / RES).astype(np.int32)
        iy = ((Q[:, 1] - y0) / RES).astype(np.int32)
        iz = ((Q[:, 2] - z0) / RES).astype(np.int32)
        np.add.at(counts, (ix, iy, iz), 1)
    centers = (
        x0 + (np.arange(nx) + 0.5) * RES,
        y0 + (np.arange(ny) + 0.5) * RES,
        z0 + (np.arange(nz) + 0.5) * RES,
    )
    return counts, centers


def _print_slice(counts, centers, axis):
    """Text dump of the densest slice perpendicular to ``axis``."""
    cmax = max(int(counts.max()), 1)
    idx = int(np.argmax(counts.sum(axis=tuple(j for j in range(3) if j != axis))))
    sl = np.take(counts, idx, axis=axis)          # 2D
    other = [j for j in range(3) if j != axis]
    print(f"--- densest {_AXES[axis]}-slice at {_AXES[axis]}="
          f"{centers[axis][idx]:.2f} ({_AXES[other[0]]} x {_AXES[other[1]]}) ---")
    for r in range(sl.shape[0]):
        print(" ".join("." if sl[r, c] == 0
                       else str(1 + int(8 * sl[r, c] / cmax))
                       for c in range(sl.shape[1])))


def main():
    headless = bool(os.environ.get("ONE_HEADLESS"))
    robot = l1.L1O6()
    poe, ch = make_poe(robot)
    counts, centers = build_grid(poe, ch)
    cmax = max(int(counts.max()), 1)
    reach = int((counts > 0).sum())
    print(f"box x{X_RANGE} y{Y_RANGE} z{Z_RANGE}, res {RES} m, "
          f"{N_SAMPLES} FK samples")
    print(f"voxels: {counts.size} total, {reach} reachable; max density {cmax}")
    for axis in range(3):
        _print_slice(counts, centers, axis)
    if headless:
        return

    import pyglet.window.key as key
    import one.viewer.world as ovw
    base = ovw.World(cam_pos=(1.9, -0.6, 1.4), cam_lookat_pos=(0.25, 0.3, 1.05))
    builtins.base = base
    ossop.frame().attach_to(base.scene)
    robot.alpha = 0.2
    robot.left_hand.alpha = 0.2
    robot.right_hand.alpha = 0.2
    robot.attach_to(base.scene)

    # full faint cloud of reachable voxels
    ijk = np.argwhere(counts > 0)
    vs = np.stack([centers[0][ijk[:, 0]],
                   centers[1][ijk[:, 1]],
                   centers[2][ijk[:, 2]]], axis=1).astype(np.float32)
    dens = counts[counts > 0].astype(np.float32) / cmax
    rgbs = np.stack([1.0 - dens, dens, np.zeros_like(dens)], axis=1)
    cloud = ossop.point_cloud(vs, rgbs, alpha=0.25)
    state = {"cloud_on": True, "axis": 2, "idx": [0, 0, counts.shape[2] // 2],
             "objs": []}
    cloud.attach_to(base.scene)

    def redraw():
        for o in state["objs"]:
            o.detach_from(base.scene)
        state["objs"] = []
        ax = state["axis"]
        k = int(np.clip(state["idx"][ax], 0, counts.shape[ax] - 1))
        state["idx"][ax] = k
        sl = np.take(counts, k, axis=ax)
        cell = np.argwhere(sl > 0)
        for (a, b) in cell:
            full = [0, 0, 0]
            full[ax] = k
            o0, o1 = [j for j in range(3) if j != ax]
            full[o0] = a
            full[o1] = b
            p = np.array([centers[0][full[0]], centers[1][full[1]],
                          centers[2][full[2]]], dtype=np.float32)
            f = float(sl[a, b]) / cmax
            s = ossop.sphere(pos=p, radius=0.008 + 0.014 * f,
                             rgb=(1.0 - f, f, 0.0))
            s.attach_to(base.scene)
            state["objs"].append(s)
        base.set_caption(
            f"slice {_AXES[ax]}={centers[ax][k]:.2f}  [{k}/{counts.shape[ax]-1}]"
            f"   (X/Y/Z axis | step: , . or [ ] or Up/Down | A cloud | O all)")
        base.scene.dirty = True

    redraw()

    def tick(dt):
        im = base.input_manager
        for kc, ax in ((key.X, 0), (key.Y, 1), (key.Z, 2)):
            if im.is_key_pressed_edge(kc):
                state["axis"] = ax
                redraw()
        if (im.is_key_pressed_edge(key.BRACKETLEFT)
                or im.is_key_pressed_edge(key.COMMA)
                or im.is_key_pressed_edge(key.DOWN)):
            state["idx"][state["axis"]] -= 1
            redraw()
        if (im.is_key_pressed_edge(key.BRACKETRIGHT)
                or im.is_key_pressed_edge(key.PERIOD)
                or im.is_key_pressed_edge(key.UP)):
            state["idx"][state["axis"]] += 1
            redraw()
        if im.is_key_pressed_edge(key.A):
            state["cloud_on"] = not state["cloud_on"]
            (cloud.attach_to if state["cloud_on"]
             else cloud.detach_from)(base.scene)
            base.scene.dirty = True
        if im.is_key_pressed_edge(key.O):
            for o in state["objs"]:
                o.detach_from(base.scene)
            state["objs"] = []
            if not state["cloud_on"]:
                state["cloud_on"] = True
                cloud.attach_to(base.scene)
            base.set_caption("all reachable voxels (X/Y/Z to slice)")
            base.scene.dirty = True

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == "__main__":
    main()
