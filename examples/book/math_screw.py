import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
from one import ovw, ossop

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, 0),
                 toggle_auto_cam_orbit=False)

# world frame
ossop.frame(pos=np.zeros(3), rotmat=np.eye(3)).attach_to(base.scene)

# target motion
p = np.array([0.1, 0.3, 0.4])
R = oum.rotmat_from_axangle(np.array([0, 0, 1]), np.pi / 3)

# black arrow: translation p
ossop.arrow(spos=np.zeros(3), epos=p,
            rgb=ouc.BasicColor.BLACK).attach_to(base.scene)

# dashed semi-transparent frame: after translation only
ossop.dashed_frame(pos=p, rotmat=np.eye(3),
                   alpha=ouc.ALPHA.LIGHT_SEMI).attach_to(base.scene)

# dashed opaque frame: after translation + rotation
ossop.dashed_frame(pos=p, rotmat=R,
                   alpha=ouc.ALPHA.SOLID).attach_to(base.scene)

# screw decomposition: (R, p) <=> rotate theta about line (s_axis through q_s),
# then slide d_s along s_axis (pitch p_s = d_s / theta)
rotvec = oum.rotvec_from_rotmat(R)
theta = np.linalg.norm(rotvec)
if theta < 1e-9:
    # pure translation: axis along p, q undefined, pitch infinite
    a_s = p / np.linalg.norm(p)
    q_s = np.zeros(3)
    d_s = np.linalg.norm(p)
    p_s = np.inf
else:
    a_s = rotvec / theta
    d_s = float(p @ a_s)
    p_s = d_s / theta
    p_perp = p - d_s * a_s
    q_s = 0.5 * p_perp + 0.5 / np.tan(theta / 2) * np.cross(a_s, p_perp)
print(f"theta={theta:.4f}  axis={a_s}  q_s={q_s}  slide={d_s:.4f}  pitch={p_s:.4f}")

# screw axis line through q_s along a_s
axis_color = ouc.ExtendedColor.VIOLET
axis_half = 0.4
ossop.cylinder(spos=q_s - axis_half * a_s,
               epos=q_s + axis_half * a_s,
               radius=0.002, rgb=axis_color).attach_to(base.scene)
# closest point on axis from origin
ossop.sphere(pos=q_s, radius=0.008, rgb=axis_color).attach_to(base.scene)
# slide d_s drawn on the axis (from q_s along a_s)
ossop.arrow(spos=q_s, epos=q_s + d_s * a_s,
            rgb=axis_color).attach_to(base.scene)

# rotation trajectory: one frame per 5 degrees, plus a dashed rotation circle
# in the plane perpendicular to the screw axis at each frame's height
n_steps = int(round(np.degrees(theta) / 5))
I3 = np.eye(3)
radius_circle = float(np.linalg.norm(q_s))


def dashed_circle_segs(center, normal, radius, n_dashes=24, dash_frac=0.6):
    helper = np.array([1., 0., 0.]) if abs(normal[2]) > 0.9 else np.array([0., 0., 1.])
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    segs = np.empty((n_dashes, 2, 3), dtype=np.float32)
    da = 2 * np.pi / n_dashes
    for i in range(n_dashes):
        a0 = i * da
        a1 = a0 + da * dash_frac
        segs[i, 0] = center + radius * (np.cos(a0) * u + np.sin(a0) * v)
        segs[i, 1] = center + radius * (np.cos(a1) * u + np.sin(a1) * v)
    return segs


for k in range(n_steps + 1):
    tau = k / n_steps
    R_tau = oum.rotmat_from_axangle(a_s, tau * theta)
    p_tau = (I3 - R_tau) @ q_s + tau * d_s * a_s
    ossop.frame(pos=p_tau, rotmat=R_tau,
                length_scale=0.5, alpha=0.5).attach_to(base.scene)
    center = q_s + tau * d_s * a_s
    segs = dashed_circle_segs(center, a_s, radius_circle)
    ossop.linsegs(segs, radius=0.001,
                  srgbs=axis_color, alpha=0.5).attach_to(base.scene)

base.run()
