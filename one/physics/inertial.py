import numpy as np
import one.scene.collision as osco

def inertia_box(m, hx, hy, hz):
    Ixx = 1 / 3 * m * (hy * hy + hz * hz)
    Iyy = 1 / 3 * m * (hx * hx + hz * hz)
    Izz = 1 / 3 * m * (hx * hx + hy * hy)
    return np.diag([Ixx, Iyy, Izz])


def inertia_sphere(m, r):
    I = 2 / 5 * m * r * r
    return np.diag([I, I, I])


def inertia_capsule(m, r, h):
    Vc = np.pi * r * r * (2 * h)
    Vs = 4 / 3 * np.pi * r ** 3
    mc = m * Vc / (Vc + 2 * Vs)
    ms = m * Vs / (Vc + 2 * Vs)
    Izz = 0.5 * mc * r * r + 2 * (2 / 5 * ms * r * r)
    Ixx = (1 / 12 * mc * (3 * r * r + 4 * h * h)
           + 2 * (2 / 5 * ms * r * r + ms * h * h))
    return np.diag([Ixx, Ixx, Izz])

def inertia_from_collisions(collisions, total_mass=10.0):
    vols = []
    for c in collisions:
        if isinstance(c, osco.SphereCollisionShape):
            vols.append(4/3*np.pi*c.radius**3)
        elif isinstance(c, osco.CapsuleCollisionShape):
            vols.append(
                2*np.pi*c.radius**2*c.half_length +
                4/3*np.pi*c.radius**3)
        elif isinstance(c, (osco.AABBCollisionShape,
                            osco.OBBCollisionShape)):
            hx, hy, hz = c.half_extents
            vols.append(8*hx*hy*hz)
        elif isinstance(c, osco.MeshCollisionShape):
            vs = c.geometry._vs
            hx, hy, hz = (vs.max(axis=0) - vs.min(axis=0)) * 0.5
            vols.append(8*hx*hy*hz)
        else:
            vols.append(1.0)  # fallback
    vols = np.asarray(vols)
    vols = np.maximum(vols, 1e-6)
    masses = total_mass * vols / vols.sum()
    com_total = np.zeros(3)
    for mi, c in zip(masses, collisions):
        com_total += mi * c.pos
    com_total /= total_mass
    I_total = np.zeros((3, 3))
    for mi, c in zip(masses, collisions):
        if isinstance(c, osco.SphereCollisionShape):
            I_local = inertia_sphere(mi, c.radius)
        elif isinstance(c, osco.CapsuleCollisionShape):
            I_local = inertia_capsule(mi, c.radius, c.half_length)
        elif isinstance(c, (osco.AABBCollisionShape,
                            osco.OBBCollisionShape)):
            hx, hy, hz = c.half_extents
            I_local = inertia_box(mi, hx, hy, hz)
        elif isinstance(c, osco.MeshCollisionShape):
            vs = c.geometry._vs
            hx, hy, hz = (vs.max(axis=0) - vs.min(axis=0)) * 0.5
            I_local = inertia_box(mi, hx, hy, hz)
        else:
            I_local = np.eye(3) * 1e-6
        R = c.rotmat
        I_body = R @ I_local @ R.T
        r = c.pos - com_total
        d = np.dot(r, r)
        I_shift = mi*(d*np.eye(3) - np.outer(r, r))
        I_total += I_body + I_shift
    return com_total, I_total
