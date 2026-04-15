"""Visualize Lite6 reachable workspace with self-collision filtering.

Monte-Carlo samples joint configs within limits, rejects self-collided
configs via MJCollider, colors each reachable TCP by manipulability
(Yoshikawa, sqrt(det(J J^T))).
"""
import numpy as np

import one.collider.mj_collider as ocm
import one.utils.constant as ouc
from one import ovw, ossop, xarm_lite6


def compute_jacobian_and_tcp(robot, qs):
    """Return (6x6 body-velocity jacobian in world, tcp_pos)."""
    chain = robot._chain
    jnts = chain.jnts
    n = len(jnts)
    root_tf = np.eye(4, dtype=np.float32)
    root_tf[:3, :3] = robot.rotmat
    root_tf[:3, 3] = robot.pos
    wd_lnk = np.empty((n + 1, 4, 4), dtype=np.float32)
    wd_lnk[0] = root_tf
    wd_jnt = np.empty((n, 4, 4), dtype=np.float32)
    for k in range(n):
        wd_jnt[k] = wd_lnk[k] @ jnts[k].zero_tf
        wd_lnk[k + 1] = wd_jnt[k] @ jnts[k].motion_tf(qs[k])
    tcp = wd_lnk[-1, :3, 3].copy()
    jac = np.zeros((6, n), dtype=np.float32)
    for k in range(n):
        ax = wd_jnt[k, :3, :3] @ jnts[k].ax
        p_k = wd_jnt[k, :3, 3]
        if jnts[k].jtype == ouc.JntType.REVOLUTE:
            jac[3:6, k] = ax
            jac[0:3, k] = np.cross(ax, tcp - p_k)
        elif jnts[k].jtype == ouc.JntType.PRISMATIC:
            jac[0:3, k] = ax
    return jac, tcp


def manipulability(jac):
    """Yoshikawa index: sqrt(det(J J^T))."""
    m = jac @ jac.T
    d = float(np.linalg.det(m))
    return np.sqrt(max(d, 0.0))


def sample_workspace(robot, mjc, n_samples, fixed=None, seed=0):
    rng = np.random.default_rng(seed)
    chain = robot._chain
    lo = np.asarray(chain.lmt_lo, dtype=np.float32)
    hi = np.asarray(chain.lmt_up, dtype=np.float32)
    print(f"[workspace] joint limits lo={np.round(lo, 3)} "
          f"hi={np.round(hi, 3)}")
    qs_mat = rng.uniform(lo, hi, size=(n_samples, lo.size)).astype(np.float32)
    if fixed:
        for idx, val in fixed.items():
            qs_mat[:, idx] = val
    ok_pts, ok_w = [], []
    n_bad = 0
    for i, qs in enumerate(qs_mat):
        if mjc.is_collided(qs):
            n_bad += 1
            continue
        jac, tcp = compute_jacobian_and_tcp(robot, qs)
        ok_pts.append(tcp)
        ok_w.append(manipulability(jac))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_samples}  ok={len(ok_pts)}  "
                  f"collided={n_bad}")
    return (np.asarray(ok_pts, dtype=np.float32),
            np.asarray(ok_w, dtype=np.float32))


def map_colors(values, lo_clip=None, hi_clip=None):
    """Blue -> red colormap. Low=blue, High=red."""
    v = values.astype(np.float32)
    if lo_clip is None:
        lo_clip = float(np.percentile(v, 2))
    if hi_clip is None:
        hi_clip = float(np.percentile(v, 98))
    t = np.clip((v - lo_clip) / max(hi_clip - lo_clip, 1e-9), 0.0, 1.0)
    rgbs = np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)
    return rgbs.astype(np.float32)


if __name__ == "__main__":
    base = ovw.World(cam_pos=(1.8, 1.8, 1.5),
                     cam_lookat_pos=(0.0, 0.0, 0.4))
    robot = xarm_lite6.Lite6()
    robot.attach_to(base.scene)

    # build a self-collision checker (robot-only scene)
    mjc = ocm.MJCollider()
    mjc.append(robot)
    mjc.actors = [robot]
    mjc.compile(margin=0.0)

    # full 3D envelope: all six joints free
    n_vol = 40000
    ok_pts, ok_w = sample_workspace(robot, mjc, n_vol, seed=1)
    if len(ok_pts):
        print(f"[workspace] reachable {len(ok_pts)}  "
              f"x[{ok_pts[:, 0].min():.3f},{ok_pts[:, 0].max():.3f}] "
              f"z[{ok_pts[:, 2].min():.3f},{ok_pts[:, 2].max():.3f}]  "
              f"w[min={ok_w.min():.4f} max={ok_w.max():.4f} "
              f"med={np.median(ok_w):.4f}]")

    # keep only high-manipulability points (e.g. top 50% by Yoshikawa index)
    w_thresh = float(np.percentile(ok_w, 50))
    mask = ok_w >= w_thresh
    print(f"[workspace] threshold w>={w_thresh:.4f} -> "
          f"{int(mask.sum())} / {len(ok_pts)} kept")
    keep_pts = ok_pts[mask]
    keep_w = ok_w[mask]
    keep_rgbs = map_colors(keep_w)
    ossop.point_cloud(keep_pts, keep_rgbs, alpha=0.6).attach_to(base.scene)

    robot.fk(qs=np.zeros(6, dtype=np.float32))
    ossop.frame(pos=(0, 0, 0),
                rotmat=np.eye(3, dtype=np.float32)).attach_to(base.scene)

    base.run()
