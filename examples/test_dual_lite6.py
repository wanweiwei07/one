"""Two Lite6 arms mounted on a shared base to form a dual-arm setup."""
import numpy as np

from one import oum, ouc, ovw, ossop, xarm_lite6


if __name__ == "__main__":
    base = ovw.World(cam_pos=(1.5, 1.5, 1.2),
                     cam_lookat_pos=(0.0, 0.0, 0.35))

    # shared torso/base plate at z=0
    plate = ossop.box(xyz_lengths=(0.5, 0.3, 0.02),
                      pos=(0.0, 0.0, 0.0),
                      rgb=ouc.BasicColor.GRAY)
    plate.attach_to(base.scene)

    # left arm: mounted on +Y side, tilted outward 30 deg around X
    left_rotmat = oum.rotmat_from_axangle(ouc.StandardAxis.X, np.deg2rad(-30))
    left = xarm_lite6.Lite6(pos=(0.0, 0.15, 0.01), rotmat=left_rotmat)
    left.attach_to(base.scene)

    # right arm: mounted on -Y side, mirror tilt
    right_rotmat = oum.rotmat_from_axangle(ouc.StandardAxis.X, np.deg2rad(30))
    right = xarm_lite6.Lite6(pos=(0.0, -0.15, 0.01), rotmat=right_rotmat)
    right.attach_to(base.scene)

    # simple "ready" pose for both
    ready = np.array([0.0, -0.5, 1.2, 0.0, 0.4, 0.0], dtype=np.float32)
    left.fk(qs=ready)
    right.fk(qs=ready)

    ossop.frame().attach_to(base.scene)
    base.run()
