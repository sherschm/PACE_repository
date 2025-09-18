import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import plot_response
import os
file_directory = os.path.dirname(os.path.abspath(__file__))

# --- System Model Parameters ---
m = 10  # kg
L = 2
g = 9.81

# Define control input
def u(t,x):
    force = 1.0  # Constant control input for this example
    return np.array([force])

# Dynamics function
def xdot(t, x):
    dx = np.array([[x[1]],
              [-(g/L)*sin(x[0])]])
    return dx.flatten()

# Initial conditions & simulation parameters
t_span = (0, 10)   # time range (0-> 10 seconds)
x0 = np.array([1.0, 0.0])    # initial state: x0=[y0, ydot0]

# Solve the ODE
solution = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation results and visualise
t = solution.t
th_sol = solution.y[0]
thd_sol = solution.y[1]
F = np.array([u(ti,[xi, vi])[0] for xi, vi, ti in zip(th_sol, thd_sol, t)])

# --- Generate plots of the cart simulation! ---
plot_response(th_sol, thd_sol, F, t, file_directory, filename="response_plot.png")