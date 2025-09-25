import numpy as np
import control as ct
import os
from scipy.integrate import solve_ivp
from PendulumPlotting import rot_pendulum_animator, plot_response
from DynamicsFuncs import rot_pend_dynamics_num
#Import the dynamic model functions
from PendulumModelling import M_f, N_f
file_directory = os.path.dirname(os.path.abspath(__file__))

A = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 57.144539, 0.0, 1.2051996]
])

B = [[0.], [0.], [ 1.], [0.83706176]]

C = np.eye(4)  # Identity output matrix (just for completeness)

D = np.zeros((4, 1))  # No direct feedthrough

sys_c = ct.ss(A, B, C, D)

Q=np.diag([0.1,0.01,0.01,0.01]) 
R=0.01

K, S, E = ct.lqr(A, B, Q, R)

# Closed-loop system matrix
A_cl = A - B @ K

# Closed-loop poles
poles = np.linalg.eigvals(A_cl)

print("LQR Gain K:\n", K)
print("Closed-loop poles:\n", poles)

# ---------------------------------------------------------------------
# 3) Simulate with SciPy
# ---------------------------------------------------------------------
# Control input function

def u(t,x):
    equil = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)  # [θ1, θ2, θ1d, θ2d]

    acceleration = K @ (equil - x)  # u = -K(x - x_equil)

    #acceleration = 0.0

    return np.array([acceleration]).flatten()

# Dynamics function for ODE solver
def xdot(t, x):
    control = u(t,x)
    return rot_pend_dynamics_num(x, control, M_f, N_f)

t_span = (0.0, 5.0)
t_eval = np.linspace(t_span[0], t_span[1], 400)
x0 = np.array([0.0, np.pi-0.1, 0.0, 0.0], dtype=float)

sol = solve_ivp(xdot, t_span, x0, t_eval=t_eval, method="RK45", atol=1e-9, rtol=1e-7)

F = np.array([u(tt, xx) for tt, xx in zip(sol.t, sol.y.T)])

# --- Plot results ---
plot_response(sol, F, file_directory, filename="response_plot.png")

# --- Animate results ---
rot_pendulum_animator(sol, name=os.path.join(file_directory, "rotary_pendulum_anim"))
