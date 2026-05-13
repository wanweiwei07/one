import os
import time
import json
import numpy as np
from tqdm import tqdm

import one.utils.math as oum
import one.robots.base.kine.numik as orbkn
import one.robots.base.kine.numik_sel as orbkis
from one.robots.manipulators.denso.cvr038.cvr038 import CVR038
from one.robots.manipulators.universal_robots.ur3.ur3 import UR3
from one.robots.manipulators.kawasaki.rs007l.rs007l import RS007L
from one.robots.manipulators.xarm.lite6.lite6 import Lite6
from one.robots.manipulators.franka.fr3.fr3 import FR3


def make_robot(name):
    pos = np.array([0.1, 0.3, 0.5], dtype=np.float32)
    if name == 'cbt':
        return CVR038(pos=pos)
    if name == 'ur3':
        return UR3(pos=pos)
    if name == 'rs007l':
        return RS007L(pos=pos)
    if name == 'lite6':
        return Lite6(pos=pos)
    if name == 'fr3':
        return FR3(pos=pos)
    raise ValueError(f"Unknown robot: {name}")


def make_solver(name, robot, default_solver):
    chain = robot._chain
    data_dir = os.path.join(robot.structure.res_dir, "data")
    if name == 'default':
        return default_solver
    if name == 'num':
        return orbkn.NumIKSolver(chain)
    if name == 'sel':
        return orbkis.SELIKSolver(chain, data_dir=data_dir)
    if name == 'halley':
        return orbkh.NumIKHalleySolver(chain)
    if name == 'sel_halley':
        return orbksh.SELIKHalleySolver(chain, data_dir=data_dir)
    raise ValueError(f"Unknown solver: {name}")


def rand_active_qs(chain):
    lo = np.asarray(chain.lmt_lo, dtype=np.float32)
    up = np.asarray(chain.lmt_up, dtype=np.float32)
    return lo + np.random.rand(len(lo)).astype(np.float32) * (up - lo)


nupdate = 10000
best_sol_num_list = [1]
robot_list = ['cbt', 'ur3', 'rs007l', 'lite6', 'fr3']  #  ['cbt', 'ur3', 'rs007l', 'lite6', 'fr3']
solver_list = ['sel']    # ['sel', 'halley', 'sel_halley']
json_file = "one/robots/base/kine/metrics_robot_result.jsonl"


if __name__ == '__main__':
    for name in robot_list:
        robot = make_robot(name)
        chain = robot._chain
        default_solver = robot._solver

        for solver_name in solver_list:
            robot._solver = make_solver(solver_name, robot, default_solver)
            solver_cls = robot._solver.__class__.__name__

            for best_sol_num in best_sol_num_list:
                success_num = 0
                time_list = []
                pos_err_list = []
                rot_err_list = []

                desc = f"{name}/{solver_name}({solver_cls})/k={best_sol_num}"
                for i in tqdm(range(nupdate), desc=desc):
                    jnt_values = rand_active_qs(chain)
                    robot.fk(qs=jnt_values)
                    tgt_pos = robot.gl_tcp_tf[:3, 3].copy()
                    tgt_rotmat = robot.gl_tcp_tf[:3, :3].copy()

                    tic = time.time()
                    result = robot.ik_tcp(
                        tgt_rotmat=tgt_rotmat,
                        tgt_pos=tgt_pos,
                        max_solutions=best_sol_num,
                    )
                    toc = time.time()
                    time_list.append(toc - tic)

                    if result is None or len(result) == 0:
                        continue
                    success_num += 1

                    robot.fk(qs=result[0])
                    pred_pos = robot.gl_tcp_tf[:3, 3]
                    pred_rotmat = robot.gl_tcp_tf[:3, :3]
                    pos_err, rot_err, _ = oum.diff_between_poses(
                        tgt_pos * 1000, tgt_rotmat,
                        pred_pos * 1000, pred_rotmat,
                    )
                    pos_err_list.append(pos_err)
                    rot_err_list.append(rot_err)

                time_arr = np.asarray(time_list)
                pos_arr = (np.asarray(pos_err_list)
                           if len(pos_err_list) else np.array([np.nan]))
                rot_arr = (np.asarray(rot_err_list)
                           if len(rot_err_list) else np.array([np.nan]))

                data_entry = {
                    "robot": robot.__class__.__name__,
                    "solver_tag": solver_name,
                    "solver_class": solver_cls,
                    "best_solution_number": best_sol_num,
                    "success_rate": f"{success_num / nupdate * 100:.2f}%",

                    "t_mean": f"{np.mean(time_arr) * 1000:.2f} ms",
                    "t_std": f"{np.std(time_arr) * 1000:.2f} ms",
                    "t_min": f"{np.min(time_arr) * 1000:.2f} ms",
                    "t_max": f"{np.max(time_arr) * 1000:.2f} ms",

                    'pos_err_mean': f"{np.mean(pos_arr):.2f} mm",
                    'pos_err_std': f"{np.std(pos_arr):.2f} mm",
                    'pos_err_min': f"{np.min(pos_arr):.2f} mm",
                    'pos_err_q1': f"{np.percentile(pos_arr, 25):.2f} mm",
                    'pos_err_q3': f"{np.percentile(pos_arr, 75):.2f} mm",
                    'pos_err_max': f"{np.max(pos_arr):.2f} mm",

                    'rot_err_mean': f"{np.mean(rot_arr) * 180 / np.pi:.2f} deg",
                    'rot_err_std': f"{np.std(rot_arr) * 180 / np.pi:.2f} deg",
                    'rot_err_min': f"{np.min(rot_arr) * 180 / np.pi:.2f} deg",
                    'rot_err_q1': f"{np.percentile(rot_arr, 25) * 180 / np.pi:.2f} deg",
                    'rot_err_q3': f"{np.percentile(rot_arr, 75) * 180 / np.pi:.2f} deg",
                    'rot_err_max': f"{np.max(rot_arr) * 180 / np.pi:.2f} deg",
                }

                with open(json_file, "a") as f:
                    f.write(json.dumps(data_entry) + "\n")
