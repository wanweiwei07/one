import time
import builtins
import numpy as np
from one import ouc, oum, ovw, ossop
from one.robots.humanoids.gento import Luna

base = ovw.World(cam_pos=(2.4, -1.6, 1.6), cam_lookat_pos=(0.5, -0.2, 0.7))
oframe = ossop.frame().attach_to(base.scene)
robot = Luna()
robot.attach_to(base.scene)
builtins.robot = robot  # for debug access
builtins.base = base
# bare Luna has no tcps; target the right Arm_R7 flange directly
robot.add_tcp('rgt_flange', robot.lnk('Arm_R7_Link'))

# zero-pose flange frame, turned so the forearm points down (-z)
rot0 = robot.tcp('rgt_flange').tf[:3, :3].copy()
tgt_rotmat = oum.rotmat_from_axangle(ouc.StandardAxis.X, oum.pi / 2) @ rot0

# 13 dof: 6 torso + 7 arm; targets low/far, beyond the fixed-torso arm's reach
chain = robot.chain('right_arm_torso')
prev_qs = robot.qs.copy()
n_ok, times_ms = 0, []
xs = np.linspace(0.4, 0.8, 4)
ys = np.linspace(-0.4, 0.0, 3)
zs = np.linspace(0.4, 1.0, 4)
for x in xs:
    for y in ys:
        for z in zs:
            tgt_pos = (x, y, z)
            ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat,
                        length_scale=.3).attach_to(base.scene)
            tic = time.perf_counter()
            # the 13-dof chain needs a larger iteration budget than the default
            _s = robot.ik(tgt_pos, tgt_rotmat, chain='right_arm_torso',
                          tcp='rgt_flange', max_solutions=1, max_iter=50,
                          ref_qs=chain.extract_active_qs(prev_qs))
            times_ms.append((time.perf_counter() - tic) * 1e3)
            if _s:
                n_ok += 1
                prev_qs = _s[0]
                tmp_robot = robot.clone()
                tmp_robot.fk(qs=_s[0])
                tmp_robot.alpha = 0.2
                tmp_robot.attach_to(base.scene)
n_all = len(xs) * len(ys) * len(zs)
print(f"right_arm_torso: {n_ok}/{n_all} solved, "
      f"avg {np.mean(times_ms):.1f} ms, max {np.max(times_ms):.1f} ms")
base.run()
