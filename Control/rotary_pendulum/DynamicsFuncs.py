import numpy as np
from PendulumParameters import rc, m, Ip, g, damping


# Rotation matrix function: assuming yaw-pitch rotation (Z-Y axes)
# ---------------------------------------------------------------------
# 2) Numerical dynamics (NumPy only)
# ---------------------------------------------------------------------
def dynamics_acc_ctrl_terms(x,M_f,N_f):
    """
    Modified dynamics with acceleration as control input.
    x = [θ1, θ2, θ1d, θ2d]
    Returns: M_acc (2x2), N_acc (2,), B_acc (2,1)
    """
    A = np.array([[1.0, 0.0]])  # constraint matrix
    M = np.array(M_f(x[0], x[1], x[2], x[3]), dtype=float)
    N = np.array(N_f(x[0], x[1], x[2], x[3]), dtype=float).flatten()

    D_mat = np.array([[0.0, 0.0],
                      [0.0, damping]], dtype=float)
    Damping_force = (D_mat @ np.asarray(x)[2:]).flatten()

    M_inv = np.linalg.inv(M)
    AMinvAT_inv = np.linalg.inv(A @ M_inv @ A.T)  # scalar (1x1)

    N_bar = N + Damping_force
    proj = A.T @ AMinvAT_inv @ A @ M_inv   # 2x2
    N_acc = N_bar - proj @ N_bar           # 2,
    B_acc = A.T @ AMinvAT_inv              # 2x1
    M_acc = M

    return M_acc, N_acc, B_acc

def rot_pend_dynamics_num(x, u, M_f, N_f):
    """
    First-order ODE: xdot = [θ1d, θ2d, θ1dd, θ2dd] = f(x,u)
    """
    M_a, N_a, B_a = dynamics_acc_ctrl_terms(x, M_f, N_f)
    acc = np.linalg.inv(M_a) @ (B_a.flatten() * u - N_a)  # (2,)
    return np.array([x[2], x[3], acc[0], acc[1]], dtype=float)

def f_wrapped(xu, M_f, N_f):
    x = xu[:4]
    u = xu[4]
    return rot_pend_dynamics_num(x, u, M_f, N_f)

def finite_diff_jacobian(f, x, eps=1e-6):
    """
    f: R^n -> R^m, returns vector
    x: (n,)
    returns J (m x n)
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)
    m = f0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        xp = x.copy()
        h = eps * max(1.0, abs(x[j]))
        xp[j] += h
        fp = np.asarray(f(xp), dtype=float)
        J[:, j] = (fp - f0) / h
    return J
