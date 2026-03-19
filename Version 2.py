import math


# =========================
# BASIC VECTOR OPERATIONS
# =========================

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = norm(v)
    if n == 0:
        return None
    return [v[0]/n, v[1]/n, v[2]/n]


# =========================
# GRAM-SCHMIDT
# =========================

def gram_schmidt(v1, v2):

    e1 = normalize(v1)
    if e1 is None:
        return None

    proj = dot(v2, e1)
    u2 = [
        v2[0] - proj*e1[0],
        v2[1] - proj*e1[1],
        v2[2] - proj*e1[2]
    ]

    e2 = normalize(u2)
    if e2 is None:
        return None

    e3 = cross(e1, e2)

    return e1, e2, e3


# =========================
# QUATERNIONS
# =========================

def q_mult(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]


def q_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def q_normalize(q):
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]


# =========================
# BASIS -> QUATERNION
# =========================

def basis_to_quaternion(e1, e2, e3):

    trace = e1[0] + e2[1] + e3[2]

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (e3[1] - e2[2]) / s
        y = (e1[2] - e3[0]) / s
        z = (e2[0] - e1[1]) / s
    elif e1[0] > e2[1] and e1[0] > e3[2]:
        s = math.sqrt(1.0 + e1[0] - e2[1] - e3[2]) * 2
        w = (e3[1] - e2[2]) / s
        x = 0.25 * s
        y = (e1[1] + e2[0]) / s
        z = (e1[2] + e3[0]) / s
    elif e2[1] > e3[2]:
        s = math.sqrt(1.0 + e2[1] - e1[0] - e3[2]) * 2
        w = (e1[2] - e3[0]) / s
        x = (e1[1] + e2[0]) / s
        y = 0.25 * s
        z = (e2[2] + e3[1]) / s
    else:
        s = math.sqrt(1.0 + e3[2] - e1[0] - e2[1]) * 2
        w = (e2[0] - e1[1]) / s
        x = (e1[2] + e3[0]) / s
        y = (e2[2] + e3[1]) / s
        z = 0.25 * s

    return q_normalize([w, x, y, z])


# =========================
# SLERP
# =========================

def slerp(q1, q2, t):

    dotp = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

    # clamp (only for floating-point safety)
    dotp = max(-1.0, min(1.0, dotp))

    if dotp < 0:
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]]
        dotp = -dotp

    theta = math.acos(dotp)
    sin_theta = math.sin(theta)

    if sin_theta == 0:
        return q1

    w1 = math.sin((1 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta

    return [
        w1*q1[0] + w2*q2[0],
        w1*q1[1] + w2*q2[1],
        w1*q1[2] + w2*q2[2],
        w1*q1[3] + w2*q2[3]
    ]


# =========================
# PRECOMPUTE (ONCE)
# =========================

def precompute_trajectory(vcts, frequency):

    dt = 1.0 / frequency

    pre_times = []
    pre_quats = []

    t = vcts[0][0]
    t_end = vcts[-1][0]

    seg_idx = 0

    while t <= t_end:

        while seg_idx < len(vcts)-2 and t > vcts[seg_idx+1][0]:
            seg_idx += 1

        t0 = vcts[seg_idx][0]
        t1 = vcts[seg_idx+1][0]

        tau = (t - t0) / (t1 - t0)

        e1_0 = vcts[seg_idx][1:4]
        e2_0 = vcts[seg_idx][4:7]
        e3_0 = vcts[seg_idx][7:10]

        e1_1 = vcts[seg_idx+1][1:4]
        e2_1 = vcts[seg_idx+1][4:7]
        e3_1 = vcts[seg_idx+1][7:10]

        q0 = basis_to_quaternion(e1_0, e2_0, e3_0)
        q1 = basis_to_quaternion(e1_1, e2_1, e3_1)

        q = slerp(q0, q1, tau)

        pre_times.append(t)
        pre_quats.append(q)

        t += dt

    return pre_times, pre_quats


# =========================
# RUNTIME COMPUTE
# =========================

def compute(vct1, vct2, time, pre_times, pre_quats, idx):

    basis = gram_schmidt(vct1, vct2)
    if basis is None:
        return None, None, idx

    # satellite frame -> lab
    q_sat_to_lab = basis_to_quaternion(*basis)

    # lab -> satellite
    q_current = q_conj(q_sat_to_lab)

    while idx < len(pre_times)-2 and time > pre_times[idx+1]:
        idx += 1

    q_desired = pre_quats[idx]

    return q_current, q_desired, idx


# =========================
# TEST for illustration
# =========================

if __name__ == "__main__":

    vcts = [
        [0,   1,0,0,  0,1,0,  0,0,1],
        [0.5, 0.6,0.4,0,  -0.77,0.64,0,  -0.26,-0.31,0.92],
        [6,   0.4,0,0.6,  -0.06,1,0,  0,0,1],
        [10,  0,0.6,0.4,  0,1,0,  -0.06,0,1]
    ]

    vct1 = [0, 0.6, 0.4]
    vct2 = [0.8, 0, 0.6]
    time_needed = 5.426

    frequency = 10  # Hz

    pre_times, pre_quats = precompute_trajectory(vcts, frequency)

    idx = 0

    q_cur, q_des, idx = compute(vct1, vct2, time_needed, pre_times, pre_quats, idx)

    print("Current quaternion:", q_cur)
    print("Desired quaternion:", q_des)