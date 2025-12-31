import warnings
import numpy as np
import numpy.typing as npt
from sklearn import cluster
from scipy.linalg import null_space
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

# numpy settings
np.set_printoptions(suppress=True)  # avoid scientific notation
# numpy proxy
pi = np.pi
eps = np.finfo(np.float32).eps
sin = np.sin
cos = np.cos
tan = np.tan

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


# helpers
def vec(*args):
    return np.array(args)


## rotmat
def rotmat_from_axangle(ax, angle):
    ax = np.asarray(ax, dtype=np.float32)
    if np.linalg.norm(ax) == 0:
        return np.eye(3, dtype=np.float32)
    return rotmat_from_rotvec(ax / np.linalg.norm(ax) * angle)


def rotmat_from_quat(quat):
    return Rotation.from_quat(quat).as_matrix().astype(np.float32)


def rotmat_from_normal(normal, up=(0, 0, 1)):
    """rotation matrix from normal vector,  normal is the z axis"""
    normal = np.asarray(normal, dtype=float)
    up = np.asarray(up, dtype=float)
    n = np.linalg.norm(normal)
    if n == 0:
        return np.eye(3)
    z = normal / n
    # remove z-component from up
    up_proj = up - np.dot(up, z) * z
    if np.linalg.norm(up_proj) < 1e-8:
        up_proj = np.array([1, 0, 0])
    y = up_proj / np.linalg.norm(up_proj)
    x = np.cross(y, z)
    return np.column_stack((x, y, z)).astype(np.float32)


def rotmat_from_normalandpoints(facet_normal, facet_first_pnt, facet_second_pnt):
    '''
    Compute the rotation matrix of a 3D facet using
    facet_normal and the first two points on the facet
    The function uses the concepts defined by Trimesh
    :param facet_normal: 1x3 nparray
    :param facet_first_pnt: 1x3 nparray
    :param facet_second_pnt: 1x3 nparray
    :return: 3x3 rotmat
    date: 20160624
    author: weiwei
    '''
    rotmat = np.eye(3, 3)
    rotmat[:, 2] = unit_vec(facet_normal)
    rotmat[:, 0] = unit_vec(facet_second_pnt - facet_first_pnt)
    if np.allclose(rotmat[:, 0], 0):
        warnings.warn("The provided facetpoints are the same! An autocomputed vector is used instead...")
        rotmat[:, 0] = orth_vec(rotmat[:, 2], toggle_unit=True)
    rotmat[:, 1] = np.cross(rotmat[:, 2], rotmat[:, 0])
    return rotmat


def rotmat_from_euler(ai, aj, ak, order='sxyz'):
    """
    :param ai: radian
    :param aj: radian
    :param ak: radian
    :param order:
    :return:
    author: weiwei
    date: 20190504
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[order]
    except (AttributeError, KeyError):
        _TUPLE2AXES[order]  # validation
        firstaxis, parity, repetition, frame = order
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    rotmat = np.eye(3, dtype=np.float32)
    if repetition:
        rotmat[i, i] = cj
        rotmat[i, j] = sj * si
        rotmat[i, k] = sj * ci
        rotmat[j, i] = sj * sk
        rotmat[j, j] = -cj * ss + cc
        rotmat[j, k] = -cj * cs - sc
        rotmat[k, i] = -sj * ck
        rotmat[k, j] = cj * sc + cs
        rotmat[k, k] = cj * cc - ss
    else:
        rotmat[i, i] = cj * ck
        rotmat[i, j] = sj * sc - cs
        rotmat[i, k] = sj * cc + ss
        rotmat[j, i] = cj * sk
        rotmat[j, j] = sj * ss + cc
        rotmat[j, k] = sj * cs - sc
        rotmat[k, i] = -sj
        rotmat[k, j] = cj * si
        rotmat[k, k] = cj * ci
    return rotmat


def rotmat_from_rotvec(v):
    return Rotation.from_rotvec(v).as_matrix().astype(np.float32)


def rotmat_between_vecs(v1, v2):
    """
    from v1 to v2?
    :param v1: 1-by-3 nparray
    :param v2: 1-by-3 nparray
    :return:
    author: weiwei
    date: 20191228
    """
    theta = angle_between_vecs(v1, v2)
    if np.allclose(theta, 0):
        return np.eye(3)
    if np.allclose(theta, np.pi):  # in this case, the rotation axis is arbitrary; I am using v1 for reference
        return rotmat_from_axangle(orth_vec(v1, toggle_unit=True), theta)
    _, axis = unit_vec(np.cross(v1, v2))
    return rotmat_from_axangle(axis, theta)


def rotmat_average(rotmat_list, bandwidth=10):
    """
    average a list of rotmat (3x3)
    :param rotmat_list:
    :param denoise: meanshift denoising is applied if True
    :return:
    author: weiwei
    date: 20190422
    """
    if len(rotmat_list) == 0:
        return False
    quaternion_list = []
    for rotmat in rotmat_list:
        quaternion_list.append(quat_from_rotmat(rotmat))
    quat_avg = quat_average(quaternion_list, bandwidth=bandwidth)
    rotmat_avg = rotmat_from_quat(quat_avg)
    return rotmat_avg


def slerp_rotmat(rotmat0, rotmat1, n):
    key_rots = Rotation.from_matrix((rotmat0, rotmat1))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    slerp_times = np.linspace(key_times[0], key_times[1], n)
    return slerp(slerp_times).as_matrix().astype(np.float32)


## (4,4) transformation matrix
def tfmat_from_axangle(ax, angle):
    """
    Compute the rodrigues matrix using the given axis and angle
    :param ax: (1, 3)
    :param angle: radian
    :return: (3,3)
    author: weiwei
    date: 20161220
    """
    length, unit_ax = unit_vec(ax)
    if length == 0:
        return np.eye(3, dtype=np.float32)
    a = np.cos(angle / 2.0)
    b, c, d = -unit_ax * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac), 0],
                     [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab), 0],
                     [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def tfmat_from_rotmat_pos(rotmat=None, pos=None):
    rotmat = ensure_rotmat(rotmat)
    pos = ensure_pos(pos)
    tfmat = np.eye(4)
    tfmat[:3, :3] = rotmat
    tfmat[:3, 3] = pos
    return tfmat.astype(np.float32)


def tfmat_from_rotvec(pos=np.zeros(3), rotvec=np.ones(3)):
    """
    build a (4,4) transformation matrix from position and rotation vector
    :param pos: (1,3)
    :param rotvec: (1,3)
    :return:
    author: weiwei
    date: 20200408, 20251128
    """
    angle, axis = unit_vec(rotvec, return_length=True)
    rotmat = rotmat_from_axangle(axis, angle)
    return tfmat_from_rotmat_pos(rotmat, pos)


def tfmat_inverse(tfmat):
    """
    compute the inverse of a homogeneous transformation matrix
    :param tfmat: (4,4)
    :return:
    author: weiwei
    date :20161213, 20251201
    """
    R = tfmat[:3, :3]
    t = tfmat[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def tfmat_average(tfmat_list, bandwidth=10):
    """TODO: tfmat list or (n,4,4) array
    average a list of tfmat (4,4)
    :param tfmat_list:
    :param bandwidth:
    :return:
    author: weiwei
    date: 20200109
    """
    tfmat_array = np.asarray(tfmat_list)
    pos_avg = pos_average(tfmat_array[:, :3, 3], bandwidth)
    rotmat_avg = rotmat_average(tfmat_array[:, :3, :3], bandwidth)
    return tfmat_from_rotmat_pos(rotmat_avg, pos_avg)


def tfmat_from_quat(quat):
    """
    convert a quaterion to homomat
    """
    q = np.array(quat, dtype=np.float32, copy=True)
    n = np.dot(q, q)
    if n < eps:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def transform_points_by_tfmat(tfmat, pnts):
    """
    do homotransform on a point or an array of points using pos
    :param tfmat: (4,4)
    :param pnts: (n,3)
    :return:
    author: weiwei
    date: 20161213
    """
    if tfmat.shape != (4, 4):
        raise ValueError(f"Homomat must be (4,4), got {tfmat.shape}")
    if pnts.ndim == 1:
        if pnts.shape[0] != 3:
            raise ValueError("Single point must have shape (3,).")
        pnts = pnts.reshape(1, 3)
    elif pnts.ndim == 2:
        if pnts.shape[1] != 3:
            raise ValueError("Points must have shape (N,3).")
    else:
        raise ValueError("pnts must be shape (3,) or (N,3).")
    R = tfmat[:3, :3]  # (3,3)
    t = tfmat[:3, 3]  # (3,)
    return pnts @ R.T + t


def interplate_pos_rotmat(start_pos,
                          start_rotmat,
                          goal_pos,
                          goal_rotmat,
                          granularity=.01):
    """
    :param start_info: [pos, rotmat]
    :param goal_info: [pos, rotmat]
    :param granularity
    :return: a list of 1xn nparray
    """
    len, vec = unit_vec(start_pos - goal_pos, return_length=True)
    n_steps = np.ceil(len / granularity)
    if n_steps < 2:
        n_steps = 2
    pos_list = np.linspace(start_pos, goal_pos, n_steps)
    rotmat_list = slerp_rotmat(start_rotmat, goal_rotmat, n_steps)
    return zip(pos_list, rotmat_list)


def interplate_pos_rotmat_around_circle(circle_center_pos,
                                        circle_normal_ax,
                                        radius,
                                        start_rotmat,
                                        end_rotmat,
                                        granularity=.01):
    """
    :param circle_center_pos:
    :param start_rotmat:
    :param end_rotmat:
    :param granularity: meter between two key points in the workspace
    :return:
    """
    vec = orth_vec(circle_normal_ax)
    angular_step_length = granularity / radius
    n_angular_steps = np.ceil(np.pi * 2 / angular_step_length)
    if n_angular_steps < 2:
        n_angular_steps = 2
    rotmat_list = slerp_rotmat(start_rotmat, end_rotmat, n_angular_steps)
    pos_list = []
    for angle in np.linspace(0, np.pi * 2, n_angular_steps).tolist():
        pos_list.append(np.dot(rotmat_from_axangle(circle_normal_ax, angle), vec * radius) + circle_center_pos)
    return zip(pos_list, rotmat_list)


def interpolate_vectors(start_vector, end_vector, granularity):
    """
    :param start_vector:
    :param end_vector:
    :param num_points:
    :param include_ends:
    :return:
    """
    max_diff = np.max(np.abs(end_vector - start_vector))
    num_intervals = np.ceil(max_diff / granularity).astype(int)
    if num_intervals < 2:
        num_intervals = 2
    interpolated_vectors = np.linspace(start_vector, end_vector, num_intervals)
    return interpolated_vectors


# quaternion
def quat_from_rotmat(rotmat):
    return Rotation.from_matrix(rotmat).as_quat().astype(np.float32)


def quat_from_axangle(ax, angle):
    ax = np.asarray(ax, dtype=np.float32)
    rotvec = ax / np.linalg.norm(ax) * angle
    return Rotation.from_rotvec(rotvec).as_quat().astype(np.float32)


def quat_average(quat_list, bandwidth=10):
    """
    average a list of quaternion (nx4)
    this is the full version
    :param rotmatlist:
    :param bandwidth: meanshift denoising is applied if available
    :return:
    author: weiwei
    date: 20190422
    """
    if len(quat_list) == 0:
        return False
    quaternionarray = np.array(quat_list)
    if bandwidth is not None:
        anglelist = []
        for quaternion in quat_list:
            anglelist.append([quaternion_to_axangle(quaternion)[0]])
        mt = cluster.MeanShift(bandwidth=bandwidth)
        quaternionarray = quaternionarray[np.where(mt.fit(anglelist).labels_ == 0)]
    nquat = quaternionarray.shape[0]
    weights = [1.0 / nquat] * nquat
    # Form the symmetric accumulator matrix
    accummat = np.zeros((4, 4))
    wsum = 0
    for i in range(nquat):
        q = quaternionarray[i, :]
        w_i = weights[i]
        accummat += w_i * (np.outer(q, q))  # rank 1 update
        wsum += w_i
    # scale
    accummat /= wsum
    # Get the eigenvector corresponding to largest eigen value
    quatavg = np.linalg.eigh(accummat)[1][:, -1]
    return quatavg


def quat_from_euler(ai, aj, ak, order='sxyz'):
    """
    Return quaternion from Euler angles and axis sequence.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[order.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[order]  # validation
        firstaxis, parity, repetition, frame = order
    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1
    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0
    return q


def rotvec_from_rotmat(rotmat):
    return Rotation.from_matrix(rotmat).as_rotvec()


def euler_from_quat(quat, order='sxyz'):
    return euler_from_rotmat(rotmat_from_quat(quat), order)


def euler_from_rotmat(rotmat, order='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[order.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[order]  # validation
        firstaxis, parity, repetition, frame = order
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if repetition:
        sy = np.sqrt(rotmat[i, j] * rotmat[i, j] + rotmat[i, k] * rotmat[i, k])
        if sy > eps:
            ax = np.atan2(rotmat[i, j], rotmat[i, k])
            ay = np.atan2(sy, rotmat[i, i])
            az = np.atan2(rotmat[j, i], -rotmat[k, i])
        else:
            ax = np.atan2(-rotmat[j, k], rotmat[j, j])
            ay = np.atan2(sy, rotmat[i, i])
            az = 0.0
    else:
        cy = np.sqrt(rotmat[i, i] * rotmat[i, i] + rotmat[j, i] * rotmat[j, i])
        if cy > eps:
            ax = np.atan2(rotmat[k, j], rotmat[k, k])
            ay = np.atan2(-rotmat[k, i], cy)
            az = np.atan2(rotmat[j, i], rotmat[i, i])
        else:
            ax = np.atan2(-rotmat[j, k], rotmat[j, j])
            ay = np.atan2(-rotmat[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return np.array([ax, ay, az])


def quaternion_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_conjugate(quaternion):
    """
    Return conjugate of quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """
    Return inverse of quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def quaternion_real(quaternion):
    """
    Return real part of quaternion.
    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """
    Return imaginary part of quaternion.
    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)


def slerp_quat(quat0, quat1, fraction, spin=0, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    """
    q0 = unit_vec(quat0[:4])
    q1 = unit_vec(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < eps:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = np.acos(d) + spin * np.pi
    if abs(angle) < eps:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0


def rand_quaternion():
    """
    Return uniform random unit quaternion.
    :return:
    """
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2) * r2, np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2])


def rand_rotmat():
    """
    Return uniform random rotation matrix.
    """
    return rotmat_from_quat(rand_quaternion())


def skew_symmetric(pos_vec):
    """
    compute the skew symmetric maxtix that corresponds to a cross
    :param pos_vec: 1x3 nparray
    :return: 3x3 skew symmetric matrix
    author: weiwei
    date: 20170421
    """
    return np.array([[0, -pos_vec[2], pos_vec[1]],
                     [pos_vec[2], 0, -pos_vec[0]],
                     [-pos_vec[1], pos_vec[0], 0]])


def orth_vec(vec, toggle_unit=True):
    """
    given a vector np.array([a,b,c]),
    this function computes an orthogonal wrs using np.array([b-c, -a+c, a-c])
    and then make it unit
    :param vec: (1,3)
    :return: (1,3)
    author: weiwei
    date: 20200528
    """
    a = vec[0]
    b = vec[1]
    c = vec[2]
    if toggle_unit:
        return unit_vec(np.array([b - c, -a + c, a - b]), return_length=False)
    else:
        return np.array([b - c, -a + c, a - b])


def rel_pose(pos0, rotmat0, pos1, rotmat1):
    """
    relpos of rot1, pos1 with respect to rot0 pos0
    :param rot0: 3x3 nparray
    :param pos0: 1x3 nparray
    :param rot1:
    :param pos1:
    :return:
    author: weiwei
    date: 20180811, 20240223
    """
    rel_pos = rotmat0.T @ (pos1 - pos0)
    rel_rotmat = rotmat0.T @ rotmat1
    return (rel_pos, rel_rotmat)


def regulate_angle(lowerbound, upperbound, jntangles):
    """
    change the range of armjnts to [lowerbound, upperbound]
    NOTE: upperbound-lowerbound must be multiplies of 2*np.pi or 360
    :param lowerbound
    :param upperbound
    :param jntangles: an array or a single joint angle
    :return:
    """
    if isinstance(jntangles, np.ndarray):
        rng = upperbound - lowerbound
        if rng >= 2 * np.pi:
            jntangles[jntangles < lowerbound] = jntangles[jntangles < lowerbound] % -rng + rng
            jntangles[jntangles > upperbound] = jntangles[jntangles > upperbound] % rng - rng
        else:
            raise ValueError("upperbound-lowerbound must be multiplies of 2*np.pi or 360")
        return jntangles
    else:
        rng = upperbound - lowerbound
        if rng >= 2 * np.pi:
            jntangles = jntangles % -rng + rng if jntangles < lowerbound else jntangles % rng - rng
        else:
            raise ValueError("upperbound-lowerbound must be multiplies of 2*np.pi or 360")
        return jntangles


def unit_vec(vec, axis=-1, return_length=True):
    """
    :param vec: (1,n)
    :param return_length: bool
    :return: (length, unit) or unit
    author: weiwei
    date: 20251128
    """
    vec = np.asarray(vec, dtype=np.float32)
    length = np.linalg.norm(vec, axis=axis, keepdims=True)
    unit = np.divide(vec, length, out=np.zeros_like(vec, dtype=np.float32), where=length > eps)
    length = length[0] if length.size == 1 else length
    return (length, unit) if return_length else unit


def angle_between_vecs(vec1, vec2):
    """
    :param vec1: (1,3)
    :param vec2: (1,3)
    :return:
    author: weiwei
    date: 20190504
    """
    l1, v1_u = unit_vec(vec1, return_length=True)
    l2, v2_u = unit_vec(vec2, return_length=True)
    if l1 == 0 or l2 == 0:
        raise ValueError("Zero length vector!")
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_2d_vecs(vec1, vec2):
    """
    return the angle from ved1 to vec2, with signs
    :param vec1: (1,2)
    :param vec2: (1,2)
    :return:
    author: weiwei
    date: 20210530
    """
    return np.atan2(vec2[1] * vec1[0] - vec2[0] * vec1[1], vec2[0] * vec1[0] + vec2[1] * vec1[1])


def delta_rotvec_between_rotmats(src_rotmat, tgt_rotmat):
    """
    compute the rotvec from src_rotmat to tgt_rotmat
    the following relation holds for the returned delta_rotvec
    tgt_rotmat = rotmat_from_rotvec(delta_rotvec) @ src_rotmat
    :param src_rotmat: (3,3)
    :param tgt_rotmat: (3,3)
    :return:
    author: weiwei
    date: 20240416
    """
    delta_rotmat = tgt_rotmat @ src_rotmat.T
    return Rotation.from_matrix(delta_rotmat).as_rotvec()
    # tmp_vec = np.array([delta_rotmat[2, 1] - delta_rotmat[1, 2],
    #                     delta_rotmat[0, 2] - delta_rotmat[2, 0],
    #                     delta_rotmat[1, 0] - delta_rotmat[0, 1]])
    # tmp_vec_norm = np.linalg.norm(tmp_vec)
    # tmp_trace_minus_one = np.trace(delta_rotmat) - 1.0
    # if np.isclose(tmp_vec_norm, 0.0):
    #     return np.zeros(3)
    # elif np.isclose(tmp_trace_minus_one, 0.0):
    #     return np.pi / 2.0 * np.diag(delta_rotmat)
    # else:
    #     return np.arctan2(tmp_vec_norm, tmp_trace_minus_one) / tmp_vec_norm * tmp_vec


# def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
#     """
#     compute angle*ax from src_rotmat to tgt_rotmat
#     the following relation holds for the returned delta_w
#     rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
#     :param src_rotmat: 3x3
#     :param tgt_rotmat: 3x3
#     :return:
#     author: weiwei
#     date: 20200326
#     """
#     delta_rotmat = tgt_rotmat @ src_rotmat.T
#     clipped_trace = np.clip((np.trace(delta_rotmat) - 1) / 2.0, -1.0, 1.0)
#     angle = np.arccos(clipped_trace)
#     if np.isclose(angle, 0.0):
#         return np.zeros(3)
#     else:
#         axis = np.array([delta_rotmat[2, 1] - delta_rotmat[1, 2],
#                          delta_rotmat[0, 2] - delta_rotmat[2, 0],
#                          delta_rotmat[1, 0] - delta_rotmat[0, 1]]) / (2 * np.sin(angle))
#         return angle * axis

def diff_between_poses(src_pos,
                       src_rotmat,
                       tgt_pos,
                       tgt_rotmat):
    """
    compute the error between the given tcp and tgt_tcp
    :param src_pos:
    :param src_rotmat
    :param tgt_pos: the position vector of the goal (could be a single value or a list of jntid)
    :param tgt_rotmat: the rotation matrix of the goal (could be a single value or a list of jntid)
    :return: a 1x6 nparray where the first three indicates the displacement in pos,
                the second three indictes the displacement in rotmat
    author: weiwei
    date: 20230929
    """
    delta = np.zeros(6)
    delta[0:3] = (tgt_pos - src_pos)
    delta[3:6] = delta_w_between_rotmat(src_rotmat, tgt_rotmat)
    pos_err = np.linalg.norm(delta[:3])
    rot_err = np.linalg.norm(delta[3:6])
    return pos_err, rot_err, delta


def cosine_between_vecs(v1, v2):
    l1, v1_u = unit_vec(v1, return_length=True)
    l2, v2_u = unit_vec(v2, return_length=True)
    if l1 == 0 or l2 == 0:
        raise Exception("One of the given vector is [0,0,0].")
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def axangle_between_rotmat(rotmati, rotmatj):
    deltaw = delta_w_between_rotmat(rotmati, rotmatj)
    angle = np.linalg.norm(deltaw)
    ax = deltaw / angle if isinstance(deltaw, np.ndarray) else None
    return ax, angle


def quaternion_to_axangle(quaternion):
    """
    :param quaternion:
    :return: angle (radian), axis
    author: weiwei
    date: 20190421
    """
    lim = 1e-12
    norm = np.linalg.norm(quaternion)
    angle = 0
    axis = [0, 0, 0]
    if norm > lim:
        w = quaternion[0]
        vec = quaternion[1:]
        normvec = np.linalg.norm(vec)
        angle = 2 * np.acos(w)
        axis = vec / normvec
    return angle, axis


def pos_average(pos_list, bandwidth=10):
    """
    average a list of pos_vec (1x3)
    :param pos_list:
    :param denoise: meanshift denoising is applied if True
    :return:
    author: weiwei
    date: 20190422
    """
    if len(pos_list) == 0:
        return False
    if bandwidth is not None:
        mt = cluster.MeanShift(bandwidth=bandwidth)
        pos_avg = mt.fit(pos_list).cluster_centers_[0]
        return pos_avg
    else:
        return np.array(pos_list).mean(axis=0)


def intersect_planes(p1, n1, p2, n2, tol=1e-6):
    # normalize normals
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < tol:
        raise ValueError("Planes are parallel or coincident.")
    A = np.array([n1, n2, d])
    b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0])
    x0 = np.linalg.solve(A, b)
    return x0, d  # x0 is a point on the line of intersection, d is the direction vector


def gen_2d_spiral_points(max_radius: float = .002,
                         radial_granularity: float = .0001,
                         tangential_granularity: float = .0003,
                         toggle_origin: bool = False) -> npt.NDArray:
    """
    gen spiral curve
    :param max_radius:
    :param radial_granularity:
    :param tangential_granularity:
    :param toggle_origin: include 0 or not
    :return:
    """
    # if tangential_granularity > radial_granularity * np.pi:
    #     warnings.warn("The tangential_granularity is suggested to be smaller than 3*radial_granularity!")
    r = np.arange(radial_granularity, max_radius, radial_granularity)
    t_ele = tangential_granularity / r
    t = np.cumsum(t_ele)
    x = r * np.cos(t)
    y = r * np.sin(t)
    if toggle_origin:
        x.insert(0, 0)
        y.insert(0, 0)
    return np.column_stack((x, y))


def gen_3d_spiral_points(pos: npt.NDArray = np.zeros(3),
                         rotmat: npt.NDArray = np.eye(3),
                         max_radius: float = .002,
                         radial_granularity: float = .0001,
                         tangential_granularity: float = .0003,
                         toggle_origin: bool = False) -> npt.NDArray:
    """
    gen spiral curve
    :param pos
    :param rotmat
    :param max_radius:
    :param tangential_granularity:
    :param toggle_origin: include 0 or not
    :return:
    """
    xy_spiral_points = gen_2d_spiral_points(max_radius=max_radius,
                                            radial_granularity=radial_granularity,
                                            tangential_granularity=tangential_granularity,
                                            toggle_origin=toggle_origin)
    xyz_spiral_points = np.column_stack((xy_spiral_points, np.zeros(len(xy_spiral_points))))
    return rotmat.dot(xyz_spiral_points.T).T + pos


def gen_regpoly(radius, nedges=12):
    angle_list = np.linspace(0, np.pi * 2, nedges + 1, endpoint=True)
    x_vertex = np.sin(angle_list) * radius
    y_vertex = np.cos(angle_list) * radius
    return np.column_stack((x_vertex, y_vertex))


def gen_2d_isosceles_verts(nlevel, edge_length, nedges=12):
    xy_array = np.asarray([[0, 0]])
    for level in range(nlevel):
        xy_vertex = gen_regpoly(radius=edge_length * (level + 1), nedges=nedges)
        for i in range(nedges):
            xy_array = np.append(xy_array,
                                 np.linspace(xy_vertex[i, :], xy_vertex[i + 1, :], num=level + 1, endpoint=False),
                                 axis=0)
    return xy_array


def gen_2d_equilateral_verts(nlevel, edge_length):
    return gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=6)


def gen_3d_isosceles_verts(pos, rotmat, nlevel=5, edge_length=0.001, nedges=12):
    xy_array = gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=nedges)
    xyz_array = np.pad(xy_array, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    return rotmat.dot((xyz_array).T).T + pos


def gen_3d_equilateral_verts(pos, rotmat, nlevel=5, edge_length=0.001):
    return gen_3d_isosceles_verts(pos=pos, rotmat=rotmat, nlevel=nlevel, edge_length=edge_length, nedges=6)


def get_aabb(pointsarray):
    """
    get the axis aligned bounding box of nx3 array
    :param pointsarray: nx3 array
    :return: center + np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
    author: weiwei
    date: 20191229
    """
    xmax = np.max(pointsarray[:, 0])
    xmin = np.min(pointsarray[:, 0])
    ymax = np.max(pointsarray[:, 1])
    ymin = np.min(pointsarray[:, 1])
    zmax = np.max(pointsarray[:, 2])
    zmin = np.min(pointsarray[:, 2])
    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    # volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    return [center, np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])]


def compute_pca(nparray):
    """
    :param nparray: nxd array, d is the dimension
    :return: evs eigenvalues, axes_mat dxn array, each column is an eigenvector
    author: weiwei
    date: 20200701osaka
    """
    ca = np.cov(nparray, y=None, rowvar=False, bias=True)  # rowvar row=point, bias biased covariance
    pcv, pcaxmat = np.linalg.eig(ca)
    return pcv, pcaxmat


def transform_data_pcv(data, random_rot=True):
    """
    :param data:
    :param random_rot:
    :return:
    author: reuishuang
    date: 20210706
    """
    pcv, pcaxmat = compute_pca(data)
    inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    x_v = pcaxmat[:, inx[2]]
    y_v = pcaxmat[:, inx[1]]
    z_v = pcaxmat[:, inx[0]]
    pcaxmat = np.asarray([y_v, x_v, -z_v]).T
    if random_rot:
        pcaxmat = np.dot(rotmat_from_axangle([1, 0, 0], np.radians(5)), pcaxmat)
        pcaxmat = np.dot(rotmat_from_axangle([0, 1, 0], np.radians(5)), pcaxmat)
        pcaxmat = np.dot(rotmat_from_axangle([0, 0, 1], np.radians(5)), pcaxmat)
    transformed_data = np.dot(pcaxmat.T, data.T).T
    return transformed_data, pcaxmat


def fit_plane(points):
    """
    :param points: nx3 nparray
    :return:
    """
    plane_center = points.mean(axis=0)
    result = np.linalg.svd(points - plane_center)
    plane_normal = unit_vec(np.cross(result[2][0], result[2][1]))
    return plane_center, plane_normal


def project_point_to_plane(point, plane_center, plane_normal):
    dist = abs((point - plane_center).dot(plane_normal))
    # print((point - plane_center).dot(plane_normal))
    if (point - plane_center).dot(plane_normal) < 0:
        plane_normal = - plane_normal
    projected_point = point - dist * plane_normal
    return projected_point


def project_vector_to_vector(vector1, vector2):
    return (vector1 @ vector2) * vector2 / (vector2 @ vector2)


def distance_point_to_edge(point, edge_start, edge_end):
    """
    compute the minimum distance from a point to a line segment.
    :param point: ndarray of the point coordinates.
    :param edge_start: ndarray of the starting point of the segment.
    :param edge_end: ndarray of the end point of the segment.
    :return: minimum distance from the point to the line segment, and the projection point
    """
    edge_vector = edge_end - edge_start
    point_vector = point - edge_start
    segment_length_squared = np.dot(edge_vector, edge_vector)
    if segment_length_squared == 0:
        return np.linalg.norm(point_vector)
    t = max(0, min(1, np.dot(point_vector, edge_vector) / segment_length_squared))
    projection = edge_start + t * edge_vector
    return np.linalg.norm(point - projection), projection


def min_distance_point_edge_list(contact_point, edge_list):
    """
    compute the minimum distance between a point and a list of nested edge lists.
    :param contact_point: (n,3)
    :param edge_list: [[edge0_v0, edge0_v1], [edge1_v0, edge1_v1], ...]
    :return: the minimum distance and projection point
    """
    min_distance = float('inf')
    min_projetion = np.zeros(3)
    for edge in edge_list:
        edge_start, edge_end = edge[0], edge[1]  # Assuming edge is a tuple/list of two vertices
        distance, projection = distance_point_to_edge(contact_point, edge_start, edge_end)
        if distance < min_distance:
            min_distance = distance
            min_projetion = projection
    return min_distance, min_projetion


def points_obb(pointsarray, toggledebug=False):
    """
    applicable to both 2d and 3d pointsarray
    :param pointsarray: nx3 or nx3 array
    :return: center, corners, and [x, y, ...] frame
    author: weiwei
    date: 20191229, 20200701osaka
    """
    pcv, pcaxmat = compute_pca(pointsarray)
    pcaxmat_t = pcaxmat.T
    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    ar = np.dot(pointsarray, np.linalg.inv(pcaxmat_t))
    # get the minimum and maximum
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    diff = (maxa - mina) * 0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    # get the corners by subtracting and adding half the bounding boxes height and width to the center
    if pointsarray.shape[1] == 2:
        corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]],
                            center + [diff[0], diff[1]], center + [-diff[0], diff[1]]])
    elif pointsarray.shape[1] == 3:
        corners = np.array([center + [-diff[0], -diff[1], -diff[2]], center + [diff[0], -diff[1], -diff[2]],
                            center + [diff[0], diff[1], -diff[2]], center + [-diff[0], diff[1], -diff[2]],
                            center + [-diff[0], diff[1], diff[2]], center + [-diff[0], -diff[1], diff[2]],
                            center + [diff[0], -diff[1], diff[2]], center + [diff[0], diff[1], diff[2]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners, pcaxmat_t)
    center = np.dot(center, pcaxmat_t)
    if toggledebug:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.scatter(pointsarray[:, 0], pointsarray[:, 1])
        ax.scatter([center[0]], [center[1]])
        ax.plot(corners[:, 0], corners[:, 1], '-')
        plt.axis('equal')
        plt.show()
    return [center, corners, pcaxmat]


def gaussian_ellipsoid(pointsarray):
    """
    compute a 95% percent ellipsoid axes_mat for the given points array
    :param pointsarray:
    :return:
    author: weiwei
    date: 20200701
    """
    pcv, pcaxmat = compute_pca(pointsarray)
    center = np.mean(pointsarray, axis=0)
    axmat = np.eye(3)
    # TODO is there a better way to do this?
    axmat[:, 0] = 2 * np.sqrt(5.991 * pcv[0]) * pcaxmat[:, 0]
    axmat[:, 1] = 2 * np.sqrt(5.991 * pcv[1]) * pcaxmat[:, 1]
    axmat[:, 2] = 2 * np.sqrt(5.991 * pcv[2]) * pcaxmat[:, 2]
    return center, axmat


def random_rgba(toggle_alpha_random=False):
    """
    randomize a 1x4 list in range 0-1
    :param toggle_alpha_random: alpha = 1 if False
    :return:
    """
    if not toggle_alpha_random:
        return np.random.random_sample(3).tolist() + [1]
    else:
        return np.random.random_sample(4).tolist()


def consecutive(nparray1d, stepsize=1):
    """
    find consecutive sequences from an array
    example:
    a = np.array([0, 47, 48, 49, 50, 97, 98, 99])
    consecutive(a)
    returns [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])]
    :param nparray1d:
    :param stepsize:
    :return:
    """
    return np.split(nparray1d, np.where(np.diff(nparray1d) != stepsize)[0] + 1)


def null_space(npmat):
    return null_space(npmat)


def to_homogeneous(pos):
    """
    append 1 to pos
    :param pos:
    :return:
    """
    return np.r_[pos, 1.0]


def reflection_homomat(point, normal):
    """
    Return matrix to mirror at plane defined by point and normal vector.
    """
    normal = unit_vec(normal[:3])
    homomat = np.identity(4)
    homomat[:3, :3] -= 2.0 * np.outer(normal, normal)
    homomat[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return homomat


def reflection_from_homomat(homomat):
    """
    Return mirror plane point and normal vector from reflection homomat
    """
    homomat = np.array(homomat, dtype=np.float64, copy=False)
    # normal: unit eigenvector corresponding to eigenvalue -1
    w, v = np.linalg.eig(homomat[:3, :3])
    i = np.where(abs(np.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = np.real(v[:, i[0]]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    w, v = np.linalg.eig(homomat)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(v[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def projection_homomat(point, normal, perspective=None, pseudo=False):
    """
    Return matrix to project onto plane defined by point and normal.
    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).
    """
    homomat = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = unit_vec(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64,
                               copy=False)
        homomat[0, 0] = homomat[1, 1] = homomat[2, 2] = np.dot(perspective - point, normal)
        homomat[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            homomat[:3, :3] -= np.outer(normal, normal)
            homomat[:3, 3] = np.dot(point, normal) * (perspective + normal)
        else:
            homomat[:3, 3] = np.dot(point, normal) * perspective
        homomat[3, :3] = -normal
        homomat[3, 3] = np.dot(perspective, normal)
    else:
        # orthogonal projection
        homomat[:3, :3] -= np.outer(normal, normal)
        homomat[:3, 3] = np.dot(point, normal) * normal
    return homomat


def affine_matrix_from_points(v0, v1, shear=True, scale=True, use_svd=True):
    """
    Return affine transform matrix to register two point sets.
    v0 and v1 are shape (n_dims, ) arrays of at least n_dims non-homogeneous
    coordinates, where n_dims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If use_svd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if n_dims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling(if specified).
    v1 = return_value @ v0
    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)
    n_dims = v0.shape[0]
    if n_dims < 2 or v0.shape[1] < n_dims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")
    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(n_dims + 1)
    M0[:n_dims, n_dims] = t0
    v0 += t0.reshape(n_dims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(n_dims + 1)
    M1[:n_dims, n_dims] = t1
    v1 += t1.reshape(n_dims, 1)
    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:n_dims].T
        B = vh[:n_dims]
        C = vh[n_dims:2 * n_dims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((n_dims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * n_dims) + (1.0,)))
    elif use_svd or n_dims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, n_dims - 1], vh[n_dims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(n_dims + 1)
        M[:n_dims, :n_dims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx + yy + zz, 0.0, 0.0, 0.0],
             [yz - zy, xx - yy - zz, 0.0, 0.0],
             [zx - xz, xy + yx, yy - xx - zz, 0.0],
             [xy - yx, zx + xz, yz + zy, zz - xx - yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= np.linalg.norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = tfmat_from_quat(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:n_dims, :n_dims] *= np.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[n_dims, n_dims]
    return M


def rotmat_from_look_at(pos, look_at, up):
    """
    Constructs a camera rotation matrix from pos to look_at
    Rotmat columns: right, up, -forward.
    """
    pos = np.asarray(pos, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    forward = look_at - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, forward)
    return np.column_stack((right, up2, -forward)).astype(np.float32)


def area_weighted_pca(verts, faces, eps=eps):
    """area weighted pca for a mesh defined by verts and faces"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # (M,3)
    cross = np.cross(v1 - v0, v2 - v0)  # (M,3)
    areas = np.linalg.norm(cross, axis=1) * 0.5  # (M,)
    total_area = areas.sum()
    if total_area < eps:
        return verts.mean(axis=0), np.array([0., 0., 1.])
    mean = (centroids * areas[:, None]).sum(axis=0) / total_area
    diff = centroids - mean  # (M,3)
    cov = (areas[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
    cov = cov / total_area
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    return mean, eig_vecs.astype(np.float32)


def ensure_right_handed(rotmat):
    """ensure the given rotmat is right-handed"""
    if np.linalg.det(rotmat) < 0:
        rotmat[:, 2] *= -1
    return rotmat.astype(np.float32)


def ensure_rotmat(rotmat=None):
    if rotmat is None:
        return np.eye(3, dtype=np.float32)
    rotmat = np.asarray(rotmat, dtype=np.float32)
    assert rotmat.shape == (3, 3)
    return rotmat


def ensure_pos(pos=None):
    if pos is None:
        return np.zeros(3, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float32)
    assert pos.shape == (3,)
    return pos


def ensure_tfmat(tfmat=None):
    if tfmat is None:
        return np.eye(4, dtype=np.float32)
    tfmat = np.asarray(tfmat, dtype=np.float32)
    assert tfmat.shape == (4, 4)
    return tfmat


def ensure_rgb(rgb=None):  # TODO make an independent util file
    from one.utils.constant import BasicColor
    if rgb is None:
        return BasicColor.DEFAULT
    rgb = np.asarray(rgb, dtype=np.float32)
    assert rgb.shape == (3,)
    return rgb
