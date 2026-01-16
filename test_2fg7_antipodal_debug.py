import builtins
from one import ovw, ouc, osso, ossop, or_2fg7, ocm
import one.physics.mj_contact as mjc
from one.grasp.antipodal import antipodal_iter

base = ovw.World(cam_pos=(.5, .5, .5),
                 cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
builtins.base = base
ossop.gen_frame().attach_to(base.scene)
gripper = or_2fg7.OR2FG7()
# # box object
# box = ossop.gen_box(half_extents=(0.015,0.015,0.015),
#                     rgb=ouc.BasicColor.ORANGE,
#                     collision_type=ouc.CollisionType.AABB,
#                     is_free=True)
# box.attach_to(base.scene)
# bunny object
bunny = osso.SceneObject.from_file(
    "bunny.stl",
    collision_type=ouc.CollisionType.MESH)
bunny.attach_to(base.scene)
bunny.toggle_render_collision = True
collider = ocm.MJCollider()
collider.append(gripper)
collider.append(bunny)
collider.actors = [gripper]
collider.compile()

# iterator
it = antipodal_iter(scene_obj=bunny,
                    gripper=gripper,
                    mj_collider=collider,
                    density=0.02,
                    normal_tol_deg=20,
                    roll_step_deg=30)
# ghost gripper + contact viz
ghost = gripper.clone()
ghost.alpha = 0.3
ghost.attach_to(base.scene)
ghost.toggle_render_collision = True
contact_viz = mjc.MJContactViz(base.scene)


def play(dt):
    try:
        pose_tf, jaw_width, score, collided = next(it)
    except StopIteration:
        return
    ghost.grip_at(pose_tf[:3, 3], pose_tf[:3, :3], jaw_width)
    if collided:
        ghost.rgb = (1.0, 0.0, 0.0)
        contact_viz.update_from_data(collider._mjenv.data)
    else:
        ghost.rgb = (0.0, 1.0, 0.0)
        contact_viz.clear()
    import numpy as np
    pos = pose_tf[:3, 3]
    rotmat = pose_tf[:3, :3]
    np.set_printoptions(precision=8, suppress=True)
    print("pos =", "np.array([" + ", ".join(f"{v:.8f}" for v in pos) + "], dtype=np.float32)")
    print("rotmat =", "np.array([" +
          ", ".join("[" + ", ".join(f"{v:.8f}" for v in row) + "]" for row in rotmat) +
          "], dtype=np.float32)")
    print("jaw_width =", f"{jaw_width:.8f}")


base.schedule_interval(play, interval=1)
base.run()
