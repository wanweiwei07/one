"""Show a Lite6 configuration with TCP reaching behind the base (negative X).

Searches random (q2, q3, q5) configs with q1=q4=q6=0 until it finds one
with TCP x<-0.3 that is also self-collision-free, then displays it.
"""
import numpy as np

import one.collider.mj_collider as ocm
from one import ovw, ossop, xarm_lite6


if __name__ == "__main__":
    base = ovw.World(cam_pos=(0.0, 2.5, 0.5),
                     cam_lookat_pos=(0.0, 0.0, 0.4))

    robot = xarm_lite6.Lite6()
    robot.attach_to(base.scene)

    mjc = ocm.MJCollider()
    mjc.append(robot)
    mjc.actors = [robot]
    mjc.compile(margin=0.0)

    chain = robot._chain
    lo = np.asarray(chain.lmt_lo, dtype=np.float32)
    hi = np.asarray(chain.lmt_up, dtype=np.float32)

    # target: TCP behind base (x<0) AND close to it (radial dist < 0.25)
    rng = np.random.default_rng(0)
    found = None
    for trial in range(50000):
        qs = rng.uniform(lo, hi).astype(np.float32)
        qs[0] = 0.0
        qs[3] = 0.0
        qs[5] = 0.0
        robot.fk(qs=qs)
        tcp = robot.gl_tcp_tf[:3, 3]
        r_xy = float(np.hypot(tcp[0], tcp[1]))
        if tcp[0] < -0.05 and r_xy < 0.25 and not mjc.is_collided(qs):
            found = (qs, tcp)
            print(f"trial {trial}: qs={np.round(qs, 3)}  "
                  f"tcp={np.round(tcp, 3)}  r_xy={r_xy:.3f}")
            break

    if found is None:
        print("no collision-free close-behind pose found.")
    else:
        qs, tcp = found
        robot.fk(qs=qs)
        ossop.frame(pos=tcp, rotmat=np.eye(3, dtype=np.float32)
                    ).attach_to(base.scene)

    ossop.frame(pos=(0, 0, 0),
                rotmat=np.eye(3, dtype=np.float32)).attach_to(base.scene)
    base.run()
