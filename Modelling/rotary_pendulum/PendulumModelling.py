import sympy as sp
import numpy as np
from sympy import Matrix, Function, symbols, simplify
from KinematicsFuncs import calc_R20
from PendulumPlotting import rot_pendulum_animator
from PendulumParameters import rc, m, Ip, g, damping
from scipy.integrate import solve_ivp
import os
from matplotlib import pyplot as plt

file_directory = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# 1) Symbolic setup
# ---------------------------------------------------------------------
t = symbols('t')
θ1 = Function('θ1')(t)
θ2 = Function('θ2')(t)
θ1d = Function('θ1d')(t)
θ2d = Function('θ2d')(t)
θ1dd = Function('θ1dd')(t)
θ2dd = Function('θ2dd')(t)
u = Function('u')(t)

vars = (t, θ1, θ2, θ1d, θ2d, θ1dd, θ2dd, u)
x = Matrix([θ1, θ2, θ1d, θ2d])

R20 = calc_R20(θ1, θ2)

subs_dict = {
    sp.Derivative(θ1, t): θ1d,
    sp.Derivative(θ2, t): θ2d,
    sp.Derivative(θ1d, t): θ1dd,
    sp.Derivative(θ2d, t): θ2dd,
}

R20_dot = R20.diff(t).doit().subs(subs_dict)

Omega = R20_dot * R20.T
omega = Matrix([
    Omega[2, 1],
    Omega[0, 2],
    Omega[1, 0]
])

# Ip can be scalar symbol or 3x3 matrix; make it 3x3 if needed
if isinstance(Ip, sp.Symbol):
    Ip = sp.eye(3) * Ip

T_rot = (sp.Rational(1, 2)) * (omega.T * Ip * omega)[0]
v_com = R20_dot.T * Matrix(rc)
T_lin = (sp.Rational(1, 2)) * m * (v_com.T * v_com)[0]
T = simplify(T_rot + T_lin)
V = simplify(m * g * (R20.T * Matrix(rc))[2])

print("Kinetic Energy T:")
sp.pprint(T, use_unicode=True)
print("\nPotential Energy V:")
sp.pprint(V, use_unicode=True)

# (Optional) numeric energy funcs
T_func = sp.lambdify((θ1, θ2, θ1d, θ2d), T, modules='numpy')
V_func = sp.lambdify((θ1, θ2, θ1d, θ2d), V, modules='numpy')

L = T - V

# Build Lagrange's Equations
dL_dθ  = [sp.diff(L, θ1), sp.diff(L, θ2)]
dL_dθd = [sp.diff(L, θ1d), sp.diff(L, θ2d)]
dt_dL_dθd = sp.Matrix(dL_dθd).diff(t).doit()

Eq = [dt_dL_dθd[i] - dL_dθ[i] for i in range(2)]
subs = {
    sp.Derivative(θ1, t): θ1d,
    sp.Derivative(θ2, t): θ2d,
    sp.Derivative(θ1d, t): θ1dd,
    sp.Derivative(θ2d, t): θ2dd
}
Eq = [sp.simplify(eq.subs(subs)) for eq in Eq]

θdd_vec = sp.Matrix([θ1dd, θ2dd])
M = sp.Matrix(Eq).jacobian(θdd_vec)
accel_vec = sp.Matrix([θ1dd, θ2dd])
N = sp.simplify(sp.expand(sp.Matrix(Eq) - M * accel_vec))

#Generate numerical functions for M and N
x_syms = [θ1, θ2, θ1d, θ2d]
M_f = sp.lambdify(x_syms, M.tolist(), modules="numpy")
N_f = sp.lambdify(x_syms, N.tolist(), modules="numpy")

# ---------------------------------------------------------------------
# 2) Numerical dynamics (NumPy only)
# ---------------------------------------------------------------------
def dynamics_acc_ctrl_terms(x):
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

def rot_pend_dynamics_num(x, u):
    """
    First-order ODE: xdot = [θ1d, θ2d, θ1dd, θ2dd]
    """
    M_a, N_a, B_a = dynamics_acc_ctrl_terms(x)
    acc = np.linalg.inv(M_a) @ (B_a.flatten() * u - N_a)  # (2,)
    return np.array([x[2], x[3], acc[0], acc[1]], dtype=float)

def f_wrapped(xu):
    x = xu[:4]
    u = xu[4]
    return rot_pend_dynamics_num(x, u)

def xdot(t, x):
    u = 0.0
    return rot_pend_dynamics_num(x, u)

# ---------------------------------------------------------------------
# 3) Simulate with SciPy
# ---------------------------------------------------------------------
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 400)
#x0 = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)
x0 = np.array([0.0, 0.1, 0.0, 0.0], dtype=float)

sol = solve_ivp(xdot, t_span, x0, t_eval=t_eval, method="RK45", atol=1e-9, rtol=1e-7)

# --- Plot results ---
plt.figure()
plt.plot(list(sol.t), list(sol.y[1]))
plt.xlabel("time (s)")
plt.ylabel("Angle (rad)")
plt.grid(True)
plt.savefig(str(file_directory)+"/response_plot.png", dpi=150)

# Depending on your animator’s API:
#   - If it accepts SciPy's OdeResult directly, keep as-is.
#   - If it wants (t, X) arrays, try passing (t_eval, sol.y.T).
rot_pendulum_animator(sol, name=os.path.join(file_directory, "rotary_pendulum_anim"))

# ---------------------------------------------------------------------
# 4) Linearization without JAX: finite-difference Jacobian
# ---------------------------------------------------------------------
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

# equilibrium and linearization
x_equil = np.array([0.0, np.pi, 0.0, 0.0, 0.0], dtype=float)  # [θ1, θ2, θ1d, θ2d, u]
J = finite_diff_jacobian(lambda z: f_wrapped(z), x_equil)     # shape (4,5)
A_matrix = J[:, :4]
B_matrix = J[:, 4:5]  # keep as column

print("Equilibrium coordinates:")
print(x_equil.tolist())
print("\nA matrix:")
print(A_matrix)
print("\nB matrix:")
print(B_matrix)
# ------------------------------------------------------------------------------
