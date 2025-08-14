import sympy as sp
import numpy as np
from sympy import Matrix, Function, symbols, simplify, diff
from KinematicsFuncs import calc_R20
from PendulumPlotting import rot_pendulum_animator
from parameters import rc,m,Ip,g,damping
from jax import jacfwd
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
import jax.numpy as jnp
from scipy.integrate import solve_ivp, odeint
import os
file_directory = os.path.dirname(os.path.abspath(__file__))

# Define time variable and functions
t = symbols('t')
θ1 = Function('θ1')(t)
θ2 = Function('θ2')(t)
θ1d = Function('θ1d')(t)
θ2d = Function('θ2d')(t)
θ1dd = Function('θ1dd')(t)
θ2dd = Function('θ2dd')(t)
u = Function('u')(t)

# Pack variables
vars = (t, θ1, θ2, θ1d, θ2d, θ1dd, θ2dd, u)
x = Matrix([θ1, θ2, θ1d, θ2d])

# Rotation matrix and derivative
R20 = calc_R20(θ1, θ2)

subs_dict = {
    sp.Derivative(θ1, t): θ1d,
    sp.Derivative(θ2, t): θ2d,
    sp.Derivative(θ1d, t): θ1dd,
    sp.Derivative(θ2d, t): θ2dd,
}

# Time derivative of R20
R20_dot = R20.diff(t).doit().subs(subs_dict)

# Angular velocity matrix
Omega = R20_dot * R20.T
omega = Matrix([
    Omega[2, 1],
    Omega[0, 2],
    Omega[1, 0]
])

# Rotational kinetic energy
# Ip is expected to be a 3x3 inertia matrix
if isinstance(Ip, sp.Symbol):
    Ip = sp.eye(3) * Ip  # default inertia matrix if not specified
T_rot = (1/2) * (omega.T * Ip * omega)[0]

# Center of mass velocity
v_com = R20_dot.T * Matrix(rc)

# Linear kinetic energy
T_lin = (1/2) * m * (v_com.T * v_com)[0]

# Total kinetic energy
T = simplify(T_rot + T_lin)

# Potential energy
V = simplify(m * g * (R20.T * Matrix(rc))[2])

# Display expressions
print("Kinetic Energy T:")
sp.pprint(T, use_unicode=True)

print("\nPotential Energy V:")
sp.pprint(V, use_unicode=True)

# Optional: create lambda functions for numerical use
T_func = sp.lambdify((θ1, θ2, θ1d, θ2d), T, modules='numpy')
V_func = sp.lambdify((θ1, θ2, θ1d, θ2d), V, modules='numpy')

L = T - V  # Lagrangian

# Partial derivatives of L
dL_dθ = [sp.diff(L, θ1), sp.diff(L, θ2)]
dL_dθd = [sp.diff(L, θ1d), sp.diff(L, θ2d)]

# Total derivatives w.r.t. time
dt_dL_dθd = sp.Matrix(dL_dθd).diff(t).doit()

# Euler-Lagrange equations
Eq = [dt_dL_dθd[i] - dL_dθ[i] for i in range(2)]

# Substitute time derivatives
subs = {
    sp.Derivative(θ1, t): θ1d,
    sp.Derivative(θ2, t): θ2d,
    sp.Derivative(θ1d, t): θ1dd,
    sp.Derivative(θ2d, t): θ2dd
}
Eq = [sp.simplify(eq.subs(subs)) for eq in Eq]
# Mass matrix extraction
θdd_vec = sp.Matrix([θ1dd, θ2dd])
M = sp.Matrix(Eq).jacobian(θdd_vec)

#M = sp.Matrix([[M11, M12], [M21, M22]])

# Compute nonlinear terms: N = Eq - M*[θ1dd; θ2dd]
accel_vec = sp.Matrix([θ1dd, θ2dd])
N = sp.simplify(sp.expand(sp.Matrix(Eq) - M * accel_vec))

x_syms = [θ1, θ2, θ1d, θ2d]
M_f = sp.lambdify(x, M.tolist(), modules="jax")
N_f = sp.lambdify(x, N.tolist(), modules="jax")

def dynamics_acc_ctrl_terms(x):

    """
    Compute modified dynamics with acceleration as control input.

    Args:
        M_f: function returning mass matrix M given state x (length-4 vector)
        N_f: function returning nonlinear vector N given state x (length-4 vector)
        x: state vector [θ1, θ2, θ1d, θ2d]
        Damping: scalar damping coefficient

    Returns:
        M_acc: Mass matrix (2×2)
        N_acc: Modified nonlinear vector (2×1)
        B_acc: Control input mapping matrix (2×1)
    """
    A = jnp.array([[1.0, 0.0]])  # constraint matrix
    M = jnp.array(M_f(x[0], x[1], x[2], x[3]))
    N = jnp.array(N_f(x[0], x[1], x[2], x[3])).flatten()

    D_mat = jnp.array([[0.0, 0.0],
                      [0.0, -damping]])
    Damping_force = (D_mat @ jnp.asarray(x)[2:]).flatten()

    M_inv = jnp.linalg.inv(M)
    AMinvAT_inv = jnp.linalg.inv(A @ M_inv @ A.T)  # scalar

    N_bar = N + Damping_force
    proj = A.T @ AMinvAT_inv @ A @ M_inv
    N_acc = N_bar - proj @ N_bar
    B_acc = A.T @ AMinvAT_inv
    M_acc = M

    return M_acc, N_acc, B_acc

# -----------------------------------------------------------------------------------
# 2. Returns dx/dt for first-order state-space dynamics
# -----------------------------------------------------------------------------------
def rot_pend_dynamics_num(x, u):
    """
    Evaluate the first-order ODE dynamics of the rotating pendulum system.

    Args:
        x: state vector [θ1, θ2, θ1d, θ2d] (length 4)
        u: control input (desired angular acceleration θ̈₁)
        M_f: function returning M(x)
        N_f: function returning N(x)
        Damping: scalar damping coefficient

    Returns:
        dxdt: derivative of state vector (length 4)
    """
    # Get modified mass matrix, nonlinear vector, input mapping
    M_a, N_a, B_a = dynamics_acc_ctrl_terms(x)

    # Compute acceleration from projected dynamics
    acc = jnp.linalg.inv(M_a) @ (B_a.flatten() * u - N_a.T)  # shape (2,)
    #print(acc)
    dxdt = [x[2], x[3], acc[0], acc[1]]
    return dxdt

def f_wrapped(xu):
    x = xu[:4]
    u = xu[4]
    return rot_pend_dynamics_num(x, u)

def xdot(t,x):
    #x = x[:4]
    u = 0.0
    return rot_pend_dynamics_num(x, u)


# Initial conditions & simulation parameters
t_span = (0, 10)   # time range (0-> 10 seconds)
x0 = jnp.array([0.0, jnp.pi, 0.0, 0.0])    # initial state: [x, xdot]

# Set up the solver
solver = Tsit5()  # A popular Runge-Kutta method
saveat = SaveAt(ts=jnp.linspace(t_span[0], t_span[1], 400))  # Save at specific time points

# Solve the ODE
sol = diffeqsolve(ODETerm(xdot), solver, t0=t_span[0], t1=t_span[1], y0=x0, saveat=saveat)

# Solve the ODE
#sol = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))
#sol = odeint(xdot, x0, np.linspace(*t_span, 400))

rot_pendulum_animator(sol, name=file_directory+"/rotary_pendulum_anim")
x_equil=[0.0, np.pi, 0.0, 0.0, 0.0]

Jacobian = jacfwd(f_wrapped)(jnp.array(x_equil))
A_matrix = jnp.stack(Jacobian, axis=0)[:4,:4]
B_matrix = jnp.stack(Jacobian, axis=0)[:,5]

print("Equilibrium coordinates:")
print(x_equil)

print("A matrix:")
print(A_matrix)

print("B matrix:")
print(B_matrix)