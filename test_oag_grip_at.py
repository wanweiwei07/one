import numpy as np
import one.utils.math as oum
from one import ovw, ossop
from one.robots.end_effectors.openarm_gripper.oa_gripper import OAGripper

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
ossop.gen_frame().attach_to(base.scene)

gripper = OAGripper()
gripper.attach_to(base.scene)

# target pose (example)
tgt_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
tgt_rotmat = oum.rotmat_from_euler(
    np.deg2rad(-45.0), 0.0, np.deg2rad(30.0))
tgt_jw = 0.04  # within [0.0, 0.088]

base_tf = gripper.grip_at(tgt_pos, tgt_rotmat, tgt_jw)

# optional: draw frames
ossop.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)
ossop.gen_frame(pos=base_tf[:3, 3],
                rotmat=base_tf[:3, :3]).attach_to(base.scene)

base.run()