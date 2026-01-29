import builtins
from one import ovw, ouc, osso, ossop, or_2fg7
from one.grasp.antipodal import antipodal_iter

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
builtins.base = base
ossop.gen_frame().attach_to(base.scene)
gripper = or_2fg7.OR2FG7()
bunny = osso.SceneObject.from_file(
    "bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.attach_to(base.scene)
# bunny.toggle_render_collision = True
it = antipodal_iter(
    gripper=gripper, tgt_sobj=bunny,
    density=0.02, normal_tol_deg=20, roll_step_deg=30)
ghost = gripper.clone()
ghost.alpha = 0.3
ghost.attach_to(base.scene)
# ghost.toggle_render_collision = True
pre_ghost = gripper.clone()
pre_ghost.rgb = (1.0, 1.0, 0.0)
pre_ghost.alpha = 0.5


def play(dt):
    try:
        pose, pre_pose, jaw_width, score, collided = next(it)
    except StopIteration:
        return
    ghost.grip_at(pose[:3, 3], pose[:3, :3], jaw_width)
    if collided:
        ghost.rgb = (1.0, 0.0, 0.0)
        pre_ghost.detach_from(base.scene)
    else:
        ghost.rgb = (0.0, 1.0, 0.0)
        pre_ghost.attach_to(base.scene)
        pre_ghost.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], jaw_width)
    # import numpy as np
    # pos = pose[:3, 3]
    # rotmat = pose[:3, :3]
    # np.set_printoptions(precision=8, suppress=True)
    # print("pos =", "np.array([" + ", ".join(f"{v:.8f}" for v in pos) + "], dtype=np.float32)")
    # print("rotmat =", "np.array([" +
    #       ", ".join("[" + ", ".join(f"{v:.8f}" for v in row) + "]" for row in rotmat) +
    #       "], dtype=np.float32)")
    # print("jaw_width =", f"{jaw_width:.8f}")


base.schedule_interval(play, interval=.1)
base.run()
