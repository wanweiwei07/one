import os
import builtins
import numpy as np

from one import oum, ouc, ovw, ossop, khi_rs007l, or_2fg7


if __name__ == "__main__":
    cand_file = "rs007l_grasp_candidates.npz"
    if not os.path.exists(cand_file):
        raise FileNotFoundError(
            "Candidate file not found. Run test_rs007l_grasp_motion.py first "
            "to generate rs007l_grasp_candidates.npz."
        )

    data = np.load(cand_file)
    pre_pos = data["pre_pos"]
    pre_rot = data["pre_rot"]
    jaw_width = data["jaw_width"]

    base = ovw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, 0.75))
    builtins.base = base
    scene = base.scene
    ossop.frame().attach_to(scene)

    robot = khi_rs007l.RS007L()
    robot.attach_to(scene)
    builtins.robot = robot

    gripper = or_2fg7.OR2FG7()
    gripper.attach_to(scene)
    robot.engage(gripper)

    ossop.frame(pos =robot.gl_tcp_tf[:3, 3],
                rotmat=robot.gl_tcp_tf[:3, :3],
                color_mat=ouc.CoordColor.MYC).attach_to(scene)

    n_total = len(pre_pos)
    n_ik = 0
    n_ik_valid = 0

    for i in range(n_total):
        tgt_pos = pre_pos[i]+np.array([0.5, 0.5, 0.3])
        tgt_rot = pre_rot[i]
        jw = float(jaw_width[i])

        qs = robot.ik_tcp_nearest(tgt_rotmat=tgt_rot, tgt_pos=tgt_pos)
        if qs is None:
            ossop.frame(
                pos=tgt_pos,
                rotmat=tgt_rot,
                color_mat=ouc.CoordColor.DYO,
                alpha=0.25,
            ).attach_to(scene)
            continue
        n_ik += 1

        ossop.frame(
            pos=tgt_pos,
            rotmat=tgt_rot,
            color_mat=ouc.CoordColor.MYC,
            alpha=0.35,
        ).attach_to(scene)

        # Optional self-check in scene only: visualize successful IK.
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.alpha = 0.08
        tmp_robot.attach_to(scene)

        # quick jaw-state consistency gate
        if gripper.jaw_range[0] <= jw <= gripper.jaw_range[1]:
            n_ik_valid += 1

    print(f"Total candidates: {n_total}")
    print(f"IK solved: {n_ik}")
    print(f"IK solved + jaw in range: {n_ik_valid}")

    base.run()
