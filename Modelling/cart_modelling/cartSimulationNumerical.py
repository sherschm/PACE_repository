import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import animate_cart, plot_response
import os
file_directory = os.path.dirname(os.path.abspath(__file__))

# --- System Model Parameters ---
m = 10  # kg
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])

# Define control input
def u(t,x):
    force = 1.0  # Constant control input for this example
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

# --- Generate plots of the cart simulation! ---
plot_response( x, xdot, F, t, file_directory, filename="response_plot.png")

# --- Generate visualisation of the cart simulation! ---
cart_anim_file = str(file_directory)+"/Cart_simulation.gif"
animate_cart(x_sol=x, F_sol=F, t_vec=t, file_name=cart_anim_file)