import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import os
from matplotlib import pyplot as plt
from PendulumPlotting import rot_pendulum_animator, plot_response
from DynamicsFuncs import finite_diff_jacobian, rot_pend_dynamics_num, f_wrapped

#Import the dynamic model functions
from PendulumModelling import M_f, N_f

#Import the Control gains
from ControlScript import K 

file_directory = os.path.dirname(os.path.abspath(__file__))

# Control input function
def u(t,x):
    equil = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)  # [θ1, θ2, θ1d, θ2d]
    acceleration = K @ (equil - x)  # u = -K(x - x_equil)
    return np.array([acceleration]).flatten()

# Dynamics function for ODE solver
def xdot(t, x):
    control = u(t,x)
    return rot_pend_dynamics_num(x, control, M_f, N_f)

# ---------------------------------------------------------------------
# 3) Simulate with SciPy
# ---------------------------------------------------------------------
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 400)
x0 = np.array([0.0, np.pi-0.1, 0.0, 0.0], dtype=float)

sol = solve_ivp(xdot, t_span, x0, t_eval=t_eval, method="RK45", atol=1e-9, rtol=1e-7)

F = np.array([u(tt, xx) for tt, xx in zip(sol.t, sol.y.T)])

# --- Plot results ---
plot_response(sol, F, file_directory, filename="response_plot.png")

# --- Animate results ---
rot_pendulum_animator(sol, name=os.path.join(file_directory, "rotary_pendulum_anim"))

# ---------------------------------------------------------------------
# 4) Linearise the model
# ---------------------------------------------------------------------
# equilibrium and linearization
x_equil = np.array([0.0, np.pi, 0.0, 0.0, 0.0], dtype=float)  # [θ1, θ2, θ1d, θ2d, u]
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
