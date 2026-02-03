import time
import builtins
import numpy as np
from one import ouc, oum, ovw, ossop
import one.robots.end_effectors.openarm_gripper.oa_gripper as oreeogog
import one.robots.manipulators.openarm.openarm as ormoo

base = ovw.World(cam_pos=(1.2, .575, 1.2), cam_lookat_pos=(0, 0, .4),
                 toggle_auto_cam_orbit=True)
oframe = ossop.gen_frame().attach_to(base.scene)
robot = ormoo.OpenArm()
robot.attach_to(base.scene)
builtins.robot = robot  # for debug access
builtins.base = base
robot.rgt_arm.fk((0, 0, 0, 0, 0, 0, 0))
robot.lft_arm.fk((0, 0, 0, 0, 0, 0, 0))
lft_gripper = oreeogog.OAGripper()
rgt_gripper = oreeogog.OAGripper()
lft_gripper.attach_to(base.scene)
rgt_gripper.attach_to(base.scene)
robot.rgt_arm.engage(rgt_gripper)
robot.lft_arm.engage(lft_gripper)
# robot.body.alpha=0.3

tgt_pos = (0.4, 0.2, 0.4)
tgt_rotmat = (oum.rotmat_from_axangle(ouc.StandardAxis.X, oum.pi / 2)@
              oum.rotmat_from_axangle(ouc.StandardAxis.Y, oum.pi / 2))
ossop.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)
qs_list = robot.lft_arm.ik_tcp(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
tcp_tf = robot.lft_arm.wd_tcp_tf
tcp_frame = ossop.gen_frame(rotmat=tcp_tf[:3, :3], pos=tcp_tf[:3, 3],
                            color_mat=ouc.CoordColor.MYC)
tcp_frame.attach_to(base.scene)
for qs in qs_list:
    tmp_lft_arm = robot.lft_arm.clone()
    tmp_lft_arm.fk(qs=qs)
    tmp_lft_arm.attach_to(base.scene)
base.run()

tgt_rotmat = oum.rotmat_from_euler(oum.pi, 0, 0)
results = []
xs = np.linspace(-1, 1, 15)
ys = np.linspace(-1, 1, 15)
zs = np.linspace(-1, 1, 15)
for x in xs:
    for y in ys:
        for z in zs:
            tgt_pos = (x, y, z)
            ossop.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)
            tic = time.perf_counter_ns()
            qs_list = robot.ik_tcp(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
            toc = time.perf_counter_ns()
            success = (len(qs_list) > 0)
            time_ns = toc - tic
            results.append(
                {"pos": tgt_pos, "success": success, "time_ns": time_ns,
                 "n_solutions": 0 if not success else len(qs_list)})
            if success:
                for qs in qs_list:
                    tmp_robot = robot.clone()
                    tmp_robot.fk(qs=qs)
                    tmp_robot.attach_to(base.scene)

succ = [r for r in results if r["success"]]
fail = [r for r in results if not r["success"]]
print("total:", len(results))
print("success:", len(succ))
print("fail:", len(fail))
if succ:
    times_ms = [r["time_ns"] / 1e6 for r in succ]
    avg_time = np.mean(times_ms)
    best_time = np.min(times_ms)
    worst_time = np.max(times_ms)
    best_case = succ[np.argmin(times_ms)]
    worst_case = succ[np.argmax(times_ms)]
    print("avg time (ms):", avg_time)
    print("best time (ms):", best_time)
    print("worst time (ms):", worst_time)
    print("avg solutions:", np.mean([r["n_solutions"] for r in succ]))
    print("\n--- BEST CASE ---")
    print("  pos:", best_case["pos"])
    print("  time (ms):", best_time)
    print("  n_solutions:", best_case["n_solutions"])
    print("\n--- WORST CASE ---")
    print("  pos:", worst_case["pos"])
    print("  time (ms):", worst_time)
    print("  n_solutions:", worst_case["n_solutions"])
base.run()
