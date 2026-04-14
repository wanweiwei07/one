import mujoco
import numpy as np
import one.physics.mj_env as mj
import pyglet.window.key as key
from one import ovw, xyt, ossop, ouc

base = ovw.World(cam_pos=(3, 1, 2), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=False)
# oframe = ossop.gen_frame()
# oframe.attach_to(base.scene)
plane_bottom = ossop.plane()
plane_bottom.attach_to(base.scene)
base_box = ossop.box(name="platform",
                     half_extents=(.5, .5, .5),
                     pos=(0, 0, .5),
                     collision_type=ouc.CollisionType.AABB,
                     is_free=False)
base_box.rgb = ouc.ExtendedColor.IVORY
base_box.toggle_render_collision = True
base_box.attach_to(base.scene)
xyt_bot = xyt.XYThetaRobot()
xyt_bot.is_free=False
xyt_bot.rgb = ouc.ExtendedColor.LAWN_GREEN
xyt_bot.attach_to(base.scene)
xyt_bot.set_rotmat_pos(pos=(0, 0, 1.11))
xyt_bot.toggle_render_collision = True

obstacle = ossop.box(half_extents=(.1, .1, .1),
                     collision_type=ouc.CollisionType.AABB,
                     mass=0.1,
                     is_free=True)
obstacle.rgb = ouc.ExtendedColor.CHOCOLATE
obstacle.toggle_render_collision = True
for i in range(5):
    obstacle_i = obstacle.clone()
    xy = np.random.uniform(-0.5, 0.5, 2)
    obstacle_i.pos = (xy[0], xy[1], i * 0.3 + 1.5)
    obstacle_i.attach_to(base.scene)

mjenv = mj.MJEnv(scene=base.scene,
                 require_ctrl=True)
mjenv.save("scene.xml")

base.schedule_interval(mjenv.step)
base.stop_after(mjenv.step, 2)


def control(dt, base, mjenv):
    k = base.input_manager.pressed_keys
    v_body = np.zeros(3)
    if key.W in k: v_body[0] += 0.5  # forward
    if key.S in k: v_body[0] -= 0.5  # backward
    if key.A in k: v_body[1] += 0.5  # left (strafe)
    if key.D in k: v_body[1] -= 0.5  # right (strafe)
    if key.Q in k: v_body[2] -= 1.0  # yaw CW
    if key.E in k: v_body[2] += 1.0  # yaw CCW
    theta = mjenv.data.qpos[2]
    c = np.cos(theta)
    s = np.sin(theta)
    # body → world Jacobian
    dq_world = np.zeros(3)
    dq_world[0] = c * v_body[0] - s * v_body[1]
    dq_world[1] = s * v_body[0] + c * v_body[1]
    dq_world[2] = v_body[2]
    dq_world *= dt
    mjenv.data.ctrl[0] += dq_world[0]
    mjenv.data.ctrl[1] += dq_world[1]
    mjenv.data.ctrl[2] += dq_world[2]
    mjenv.step(dt)


base.schedule_interval(control, .01, base, mjenv)
base.run()
