import builtins
import numpy as np
from one import rm, wd, khi_rs007l, prims, spdr, rrt

base = wd.World(cam_pos=(.75, 2, 2), cam_lookat_pos=(0, 0, 0.5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe = prims.gen_frame()
oframe.attach_to(base.scene)
robot = khi_rs007l.RS007L()
robot.attach_to(base.scene)


def is_state_valid(state):
    return True


jlmt_low = robot.structure.compiled.jlmt_low_by_idx
jlmt_high = robot.structure.compiled.jlmt_high_by_idx
sspp = spdr.SpaceProvider.from_box_bounds(lmt_low=jlmt_low,
                                          lmt_high=jlmt_high,
                                          is_state_valid=is_state_valid)
planner = rrt.RRTConnectPlanner(ssp_provider=sspp, step_size=np.pi/36)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-rm.pi / 2, -rm.pi / 4, rm.pi / 2, -rm.pi / 2, rm.pi / 4, rm.pi / 3])
state_list = planner.solve(start=start, goal=goal)
robot1 = robot.clone()
robot1.fk(qs=start)
robot1.attach_to(base.scene)
robot2 = robot.clone()
robot2.fk(qs=goal)
robot2.attach_to(base.scene)
counter = [0]

def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0
base.schedule_interval(update_pose, interval=0.1, counter=counter)

import one.physics.mj_env as mj
mjenv = mj.MjEnv(scene=base.scene)
mjenv.sync_mechstates_to_mujoco_dynamics()
mjenv.save_xml("scene.xml")
base.schedule_interval(mjenv.step)
base.run()
