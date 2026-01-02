import time
import builtins
import numpy as np
from one import rm, wd, prims, khi_rs007l

base = wd.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=True)
oframe = prims.gen_frame().attach_to(base.scene)
robot = khi_rs007l.RS007L(base_rotmat=rm.rotmat_from_euler(0,0,-rm.pi/2))
print((robot._solver.lmt_low + robot._solver.lmt_up) * 0.5)
robot.fk(qs=(robot._solver.lmt_low + robot._solver.lmt_up) * 0.5)
robot.attach_to(base.scene)
builtins.robot = robot  # for debug access
builtins.base = base

tgt_rotmat = rm.rotmat_from_euler(rm.pi, 0, 0)
results = []
xs = np.linspace(-1, 1, 15)
ys = np.linspace(-1, 1, 15)
zs = np.linspace(-1, 1, 15)
for x in xs:
    for y in ys:
        for z in zs:
            tgt_pos = (x, y, z)
            prims.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)
            tic = time.perf_counter_ns()
            qs, _ = robot.ik_tcp(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
            toc = time.perf_counter_ns()
            success = qs is not None
            results.append({
                "pos": tgt_pos,
                "success": success,
                "time_ns": toc - tic,
                "qs": qs
            })
            if success:
                tmp_robot = robot.clone()
                tmp_robot.fk(qs=qs)
                tmp_robot.attach_to(base.scene)

succ = [r for r in results if r["success"]]
fail = [r for r in results if not r["success"]]
print("success:", len(succ))
print("fail:", len(fail))
print("avg time (ms):",
      np.mean([r["time_ns"] for r in succ]) / 1e6)

base.run()
