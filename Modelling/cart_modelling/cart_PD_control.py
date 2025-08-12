import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cart_plotting import make_animation_with_force
import os
file_directory = os.path.dirname(os.path.abspath(__file__))


# --- System Parameters ---
m = 10  # kg
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])

# Desired motion
cmnd = 7.0      # desired position
cmnd_dot = 0.0  # desired velocity

# PD gains
K_p, K_d = 100, 60

# Control input function
def u(q, t):
    return K_p * (cmnd - q[0]) + K_d * (cmnd_dot - q[1])

# Dynamics function
def qdot(t, q):
    return (A @ q + B.flatten() * u(q, t))

# Initial conditions & simulation parameters
t_span = (0, 10)   # time range
q0 = [0.0, 0.0]    # initial state: [x, xdot]

# Solve the ODE
solution = solve_ivp(qdot, t_span, q0, t_eval=np.linspace(*t_span, 400))

# Extract data
t = solution.t
x = solution.y[0]
xdot = solution.y[1]
F_vec = np.array([u([xi, vi], ti) for xi, vi, ti in zip(x, xdot, t)])

# --- Plot results ---
plt.figure()
plt.plot(t, x, label="x (m)")
plt.plot(t, xdot, label="dx/dt (m/s)")
plt.xlabel("time (s)")
plt.legend()
plt.grid(True)
plt.savefig("response_plot.png", dpi=150)

make_animation_with_force(x_sol=x, F_sol=F_vec, t_vec=t, file_name=str(file_directory)+"/Cart_simulation.gif")
