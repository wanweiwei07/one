import builtins
import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.robots.manipulators.kawasaki.rs007l.rs007l as ormkr7
import one.robots.end_effectors.onrobot.or_2fg7.or_2fg7 as oreo2fg7
import one.robots.end_effectors.onrobot.or_sd.or_sd as oreorsd


if __name__ == '__main__':
    base = ovw.World(cam_pos=(1.9, 1.3, 1.3), cam_lookat_pos=(0.2, 0.0, 0.35))
    builtins.base = base
    scene = base.scene
    ossop.frame().attach_to(scene)

    # tool z-axis points straight down (approach / stand orientation)
    DOWN = oum.rotmat_from_euler(oum.pi, 0.0, 0.0)

    # ---- robot (starts bare, no end-effector) ----
    robot = ormkr7.RS007L()
    robot.attach_to(scene)
    builtins.robot = robot
    home_qs = robot.qs.copy()
    chain = robot.chain('main')
    flange = robot.runtime_lnks[-1]

    # ---- tool stands (a stand pose IS a flange pose; the EE rests there with
    # its root coincident with the flange, so picking it up is a no-jump mount) ----
    FG7_STAND = np.array([0.0, 0.45, 0.45], dtype=np.float32)
    SD_STAND = np.array([0.0, -0.45, 0.45], dtype=np.float32)

    fg7 = oreo2fg7.OR2FG7()
    fg7.set_pos_rotmat(pos=FG7_STAND, rotmat=DOWN)
    fg7.attach_to(scene)

    screwdriver = oreorsd.ORSD()
    screwdriver.set_pos_rotmat(pos=SD_STAND, rotmat=DOWN)
    screwdriver.attach_to(scene)

    for p in (FG7_STAND, SD_STAND):
        ossop.frame(pos=p, rotmat=DOWN, color_mat=ouc.CoordColor.DYO).attach_to(scene)

    # ---- bunny + place target ----
    BUNNY = np.array([0.45, 0.20, 0.0], dtype=np.float32)
    PLACE = np.array([0.45, -0.20, 0.0], dtype=np.float32)
    APPROACH = np.array([0.0, 0.0, 0.12], dtype=np.float32)   # above a pick/place
    GRASP_Z = np.array([0.0, 0.0, 0.05], dtype=np.float32)    # gripper grasp height
    WORK_Z = np.array([0.0, 0.0, 0.07], dtype=np.float32)     # screwdriver work height

    bunny = osso.SceneObject.from_file(
        'bunny.stl', collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    bunny.is_floating = True
    bunny.pos = BUNNY.copy()
    bunny.attach_to(scene)

    # ---- tcp resolvers (valid only while the EE is mounted) ----
    gc = lambda: fg7.tcp('grasp_center')
    sd = lambda: screwdriver.tcp('tip')

    # ---- operation list: ('move', pos, tcp) | ('move_qs', qs) | ('act', fn, label) ----
    ops = []
    mv = lambda pos, tcp: ops.append(('move', np.asarray(pos, dtype=np.float32), tcp))
    do = lambda fn, label: ops.append(('act', fn, label))

    # 1. pick up the 2FG7 gripper from its stand
    mv(FG7_STAND, 'flange')
    do(lambda: robot.mount(fg7, flange, update=True), 'mount 2FG7')
    # 2. grasp the bunny and place it at the target
    mv(BUNNY + APPROACH, gc)
    mv(BUNNY + GRASP_Z, gc)
    do(lambda: (fg7.close(), fg7.attach(bunny)), 'grasp bunny')
    mv(BUNNY + APPROACH, gc)
    mv(PLACE + APPROACH, gc)
    mv(PLACE + GRASP_Z, gc)
    do(lambda: (fg7.open(), fg7.detach(bunny)), 'place bunny')
    mv(PLACE + APPROACH, gc)
    # 3. return the gripper to its stand and unmount it
    mv(FG7_STAND, 'flange')
    do(lambda: robot.unmount(fg7), 'unmount 2FG7')
    # 4. pick up the screwdriver from its stand
    mv(SD_STAND, 'flange')
    do(lambda: robot.mount(screwdriver, flange, update=True), 'mount or_sd')
    # 5. work above the placed bunny
    mv(PLACE + APPROACH, sd)
    mv(PLACE + WORK_Z, sd)
    mv(PLACE + APPROACH, sd)
    # 6. return the screwdriver to its stand and unmount it
    mv(SD_STAND, 'flange')
    do(lambda: robot.unmount(screwdriver), 'unmount or_sd')
    # 7. back to the robot's initial state
    ops.append(('move_qs', home_qs))

    # ---- joint-space animation scheduler (linear interp between key configs) ----
    FRAMES = 45
    st = {'i': 0, 'f': 0, 'n': 0,
          'q0': robot.qs.copy(), 'q1': robot.qs.copy(), 'done': False}

    def _start(target_qs):
        st['q0'] = robot.qs.copy()
        st['q1'] = np.asarray(target_qs, dtype=np.float32)
        st['f'] = 0
        st['n'] = FRAMES

    def _begin():
        while st['i'] < len(ops):
            op = ops[st['i']]
            if op[0] == 'act':
                op[1]()
                print('[toolchange]', op[2])
                st['i'] += 1
                continue
            if op[0] == 'move':
                _, pos, tcp = op
                tcp_v = tcp() if callable(tcp) else tcp
                ref = chain.extract_active_qs(robot.qs)
                sols = robot.ik(pos, DOWN, tcp=tcp_v, max_solutions=1, ref_qs=ref)
                if not sols:
                    print('[toolchange] IK unreachable at op', st['i'], '- skipped')
                    st['i'] += 1
                    continue
                _start(sols[0])
                return
            # move_qs
            _start(op[1])
            return
        st['done'] = True

    _begin()

    def tick(dt):
        if st['done']:
            return
        if st['f'] >= st['n']:
            st['i'] += 1
            _begin()
            return
        st['f'] += 1
        a = st['f'] / st['n']
        a = a * a * (3.0 - 2.0 * a)   # smoothstep ease in/out
        q = (1.0 - a) * st['q0'] + a * st['q1']
        robot.fk(qs=q.astype(np.float32))

    base.schedule_interval(tick, interval=1.0 / 60.0)
    base.run()
