import sympy as sp
import numpy as np
from sympy import Matrix, Function, symbols, simplify
from PendulumParameters import rc, m, Ip, g, damping
from scipy.integrate import solve_ivp
import os
from matplotlib import pyplot as plt
from KinematicsFuncs import calc_R20
from PendulumPlotting import rot_pendulum_animator
from DynamicsFuncs import finite_diff_jacobian, rot_pend_dynamics_num, f_wrapped

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
# 4) Linearization model
# ---------------------------------------------------------------------

# equilibrium and linearization
#Choose your equilibrium point here:
x_equil = np.array([0.0, np.pi, 0.0, 0.0, 0.0], dtype=float)  # Upward vertical [θ1, θ2, θ1d, θ2d, u]
#x_equil = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)  # Downward vertical  [θ1, θ2, θ1d, θ2d, u]

J = finite_diff_jacobian(lambda z: f_wrapped(z, M_f, N_f), x_equil)     # shape (4,5)
A_matrix = J[:, :4]
B_matrix = J[:, 4:5]  # keep as column

print("Equilibrium coordinates:")
print(x_equil.tolist())
print("\nA matrix:")
print(A_matrix)
print("\nB matrix:")
print(B_matrix)
# ------------------------------------------------------------------------------
