import mujoco
import numpy as np
import one.physics.mj_env as mj
import pyglet.window.key as key
from one import ovw, xyt, ossop, ouc

base = ovw.World(cam_pos=(3, 1, 2), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=False)
# oframe = ossop.gen_frame()
# oframe.attach_to(base.scene)
plane_bottom = ossop.gen_plane()
plane_bottom.attach_to(base.scene)
base_box = ossop.gen_box(half_extents=(.5, .5, .5),
                         pos=(0, 0, .5),
                         collision_type=ouc.CollisionType.AABB,
                         is_fixed=True)
base_box.rgb = ouc.ExtendedColor.IVORY
base_box.attach_to(base.scene)
xyt_bot = xyt.XYThetaRobot()
xyt_bot.rgb = ouc.ExtendedColor.LAWN_GREEN
xyt_bot.attach_to(base.scene)
xyt_bot.base_pos = (0, 0, 1)
xyt_bot.toggle_render_collision = True

box = ossop.gen_box(half_extents=(.1, .1, .1),
                    collision_type=ouc.CollisionType.CAPSULE,
                    is_fixed=False,
                    mass=2)
box.rgb = ouc.ExtendedColor.CHOCOLATE
box.toggle_render_collision = True
for i in range(5):
    box_i = box.clone()
    xy = np.random.uniform(-0.5, 0.5, 2)
    box_i.pos = (xy[0], xy[1], i * 0.3 + 1.5)
    box_i.attach_to(base.scene)

mjenv = mj.MjEnv(scene=base.scene)
def stop(dt, function):
    base.stop(function)

base.schedule_interval(mjenv.step)
base.schedule_once(stop, 2, mjenv.step)


def control(dt, base, xyt_bot, mjenv):
    k = base.input_manager.pressed_keys
    dq = np.zeros(3)
    if key.W in k: dq[0] += 0.01
    if key.S in k: dq[0] -= 0.01
    if key.A in k: dq[1] -= 0.01
    if key.D in k: dq[1] += 0.01
    if key.Q in k: dq[2] -= 0.02
    if key.E in k: dq[2] += 0.02
    mjenv.data.ctrl[0] += dq[0]
    mjenv.data.ctrl[1] += dq[1]
    mjenv.data.ctrl[2] += dq[2]
    mjenv.step(dt)
    print(
        "ctrl=", mjenv.data.ctrl[:3],
        "qpos=", mjenv.data.qpos[:3],
        "ncon=", mjenv.data.ncon
    )
    for i in range(mjenv.data.ncon):
        c = mjenv.data.contact[i]
        g1, g2 = c.geom1, c.geom2

        b1 = mjenv.model.geom_bodyid[g1]
        b2 = mjenv.model.geom_bodyid[g2]

        body1 = mujoco.mj_id2name(mjenv.model, mujoco.mjtObj.mjOBJ_BODY, b1)
        body2 = mujoco.mj_id2name(mjenv.model, mujoco.mjtObj.mjOBJ_BODY, b2)

        print(i, body1, "<->", body2, "dist=", c.dist)

base.schedule_interval(control, .01, base, xyt_bot, mjenv)
base.run()
