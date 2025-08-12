import numpy as np
from numpy import cos, sin
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

# Define control input
def u(t,x):
    force = 15*cos(1.2*t)
    return np.array([force])

# Dynamics function
def xdot(t, x):
    dx = A @ x + B @ u(t,x)
    return dx.flatten()

# Initial conditions & simulation parameters
t_span = (0, 10)   # time range (0-> 10 seconds)
x0 = np.array([0.0, 0.0])    # initial state: [x, xdot]

# Solve the ODE
solution = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation results and visualise
t = solution.t
x = solution.y[0]
xdot = solution.y[1]
F = np.array([u(ti,[xi, vi])[0] for xi, vi, ti in zip(x, xdot, t)])

# --- Plot results ---
plt.figure()
plt.plot(t, x, label="x (m)")
plt.plot(t, xdot, label="dx/dt (m/s)")
plt.xlabel("time (s)")
plt.legend()
plt.grid(True)
plt.savefig(str(file_directory)+"/response_plot.png", dpi=150)

plt.figure()
plt.plot(t, F)
plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.savefig(str(file_directory)+"/force_plot.png", dpi=150)

# --- Generate visualisation of the cart simulation! ---
make_animation_with_force(x_sol=x, F_sol=F, t_vec=t, file_name=str(file_directory)+"/Cart_simulation.gif")