import builtins
import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
import one.motion.trajectory.cartesian as omtr
import one.robots.manipulators.kawasaki.rs007l.rs007l as ormkr7
import one.robots.end_effectors.onrobot.or_sd.or_sd as oreorsd


if __name__ == '__main__':
    base = ovw.World(cam_pos=(2.0, 0.8, 1.6), cam_lookat_pos=(0.0, 0.0, 0.7))
    builtins.base = base
    scene = base.scene
    ossop.frame().attach_to(scene)

    robot = ormkr7.RS007L()
    robot.attach_to(scene)
    builtins.robot = robot

    screwdriver = oreorsd.ORSD()
    screwdriver.attach_to(scene)
    robot.engage(screwdriver, engage_tf=oum.tf_from_rotmat_pos(pos=(0.0, 0.0, 0.05)))

    # find a reachable start tcp first (for ORSD this is stricter than 2FG7)
    start_rotmat = oum.rotmat_from_euler(oum.pi, 0.0, oum.pi)
    start_pos = np.array([0.45, -0.35, 0.15], dtype=np.float32)
    q_start = robot.ik_tcp_nearest(tgt_rotmat=start_rotmat, tgt_pos=start_pos)
    if q_start is None:
        print('Cannot find a reachable ORSD start pose.')
        base.run()
        raise SystemExit

    robot.fk(qs=q_start)
    start_rotmat = robot.gl_tcp_tf[:3, :3].copy()
    start_pos = robot.gl_tcp_tf[:3, 3].copy()
    goal_pos = start_pos + np.array([0.5, 0.5, 0.2], dtype=np.float32)
    goal_rotmat = start_rotmat.copy()

    ossop.frame(pos=start_pos, rotmat=start_rotmat, color_mat=ouc.CoordColor.DYO).attach_to(scene)
    ossop.frame(pos=goal_pos, rotmat=goal_rotmat, color_mat=ouc.CoordColor.MYC).attach_to(scene)

    q_seq, pose_seq = omtr.cartesian_to_jtraj(
        robot=robot,
        start_rotmat=start_rotmat,
        start_pos=start_pos,
        goal_rotmat=goal_rotmat,
        goal_pos=goal_pos,
        pos_step=0.01,
        rot_step=np.deg2rad(2.0),
        ref_qs=q_start,
    )
    if q_seq is None:
        print('cartesian_to_jtraj failed (IK failed on at least one sample).')
        pos_seq, rotmat_seq = pose_seq
        for pos, rotmat in zip(pos_seq, rotmat_seq):
            ossop.frame(
                pos=pos,
                rotmat=rotmat,
                color_mat=ouc.CoordColor.DYO,
                alpha=0.2,
            ).attach_to(scene)
        base.run()
    else:
        print(f'cartesian_to_jtraj success: {len(q_seq)} waypoints')
        pos_seq, rotmat_seq = pose_seq
        for pos, rotmat in zip(pos_seq, rotmat_seq):
            ossop.frame(
                pos=pos,
                rotmat=rotmat,
                color_mat=ouc.CoordColor.DYO,
                alpha=0.12,
            ).attach_to(scene)

        idx = [0]

        def tick(dt):
            robot.fk(qs=q_seq[idx[0]])
            idx[0] += 1
            if idx[0] >= len(q_seq):
                idx[0] = 0

        base.schedule_interval(tick, interval=0.03)
        base.run()
