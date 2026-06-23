import builtins
import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
import one.geom.geometry as ogg
import one.robots.manipulators.kawasaki.rs007l.rs007l as ormkr7
import one.robots.end_effectors.onrobot.or_2fg7.or_2fg7 as oreo2fg7


if __name__ == "__main__":
    base = ovw.World(cam_pos=(1.8, 1.2, 1.4), cam_lookat_pos=(0.0, 0.0, 0.5))
    builtins.base = base
    scene = base.scene
    ossop.frame().attach_to(scene)

    robot = ormkr7.RS007L(rotmat=oum.rotmat_from_euler(0, 0, -np.pi / 2))
    robot.attach_to(scene)
    builtins.robot = robot

    gripper = oreo2fg7.OR2FG7()
    gripper.set_opening(0.03)
    gripper.attach_to(scene)

    # loc_tf is flange->ee_base transform
    loc_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotmat=np.eye(3, dtype=np.float32),
    )
    robot.mount(gripper, robot.runtime_lnks[-1], loc_tf, update=True)

    tgt_pos = np.array([0.55, 0.10, 0.35], dtype=np.float32)
    ico_geom = ogg.gen_icosphere_geom(radius=1.0, n_subs=1)
    dirs = ico_geom.vs.copy()
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + oum.eps)

    n_success = 0
    for i, d in enumerate(dirs):
        # Use each icosphere vertex direction as target TCP +Z direction.
        tgt_rotmat = oum.rotmat_from_normal(d)
        ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=ouc.CoordColor.DYO, alpha=0.2).attach_to(scene)
        _s = robot.ik(tgt_pos, tgt_rotmat, tcp=gripper.tcp('grasp_center'), max_solutions=1)
        qs = _s[0] if _s else None
        if qs is None:
            continue
        n_success += 1
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.attach_to(base.scene)
        tmp_robot.alpha = .3
        ossop.frame(
            pos=tmp_robot.tcp('flange').tf[:3, 3],
            rotmat=tmp_robot.tcp('flange').tf[:3, :3],
            color_mat=ouc.CoordColor.MYC,
        ).attach_to(scene)
    print(f"IK success: {n_success}/{len(dirs)}")

    base.run()
