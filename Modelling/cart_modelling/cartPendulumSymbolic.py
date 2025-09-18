import sympy as sp
from sympy import symbols, Function, diff, Eq, simplify, pprint
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import animate_cartpole, plot_response_cartpole
import os
file_directory = os.path.dirname(os.path.abspath(__file__))
# -----------------------------
# Step 1: Define symbols
# -----------------------------
from sympy import symbols, Function, diff, sin, cos, simplify, pprint

t = symbols('t')
m, M, u, L, g = symbols('m M u L g')  # cart mass, pendulum mass, force, length, gravity

y = Function('y')(t)       # cart position
theta = Function('theta')(t)  # pendulum angle from vertical

# -----------------------------
# Step 2: Define velocities
# -----------------------------
y_dot = diff(y, t)
y_ddot = diff(y, t, t)
theta_dot = diff(theta, t)
theta_ddot = diff(theta, t, t)

# pendulum coordinates
x_p = y + L*sin(theta)
y_p = -L*cos(theta)

x_p_dot = diff(x_p, t) # pendulum horizontal velocity
y_p_dot = diff(y_p, t) # pendulum vertical velocity

# -----------------------------
# Step 3: Energies
# -----------------------------
T = (1/2)*m*y_dot**2 + (1/2)*M*(x_p_dot**2 + y_p_dot**2)
V = M*g*L*y_p

Lagr = T - V

print("Lagrangian L =")
pprint(simplify(Lagr))

# -----------------------------
# Step 4: Lagrange’s equations
# -----------------------------
# For y (cart coordinate)
dL_dydot = diff(Lagr, y_dot)
d_dt_dL_dydot = diff(dL_dydot, t)
dL_dy = diff(Lagr, y)
eq_y = simplify(d_dt_dL_dydot - dL_dy - u)   # external force u

# For theta (pendulum coordinate)
dL_dthetadot = diff(Lagr, theta_dot)
d_dt_dL_dthetadot = diff(dL_dthetadot, t)
dL_dtheta = diff(Lagr, theta)
eq_theta = simplify(d_dt_dL_dthetadot - dL_dtheta)  # no torque applied

print("\nEquation of Motion for cart (y):")
pprint(eq_y)

print("\nEquation of Motion for pendulum (theta):")
pprint(eq_theta)

# -----------------------------
# Step 6: Simulate the system!
# -----------------------------
# rearrange equations to find Mass matrix M and other terms N: M(q)*qdd = N(q,qd)
Mass_Matrix, N_vec = sp.linear_eq_to_matrix([eq_y, eq_theta], [y_ddot, theta_ddot])

print("\n Mass Matrix:")
pprint(Mass_Matrix)

print("\n Other terms N:")
pprint(N_vec)

q = [y, theta]
qd= [y_dot, theta_dot]
qdd = list(simplify(Mass_Matrix.inv())*N_vec)

# State vector and dynamics
state = sp.Matrix(q + qd)
state_dot = sp.Matrix(qd+qdd)

# Convert symbolic model to numerical function
state_dot_func = sp.lambdify((t, state, u, m, M, L, g), list(state_dot), "numpy")

# Control input (scalar force)
def u_func(t, state):
    return 0.0  # constant input

# System dynamics for solver
def system(t, x):
    u_val = float(u_func(t, x))
    xdot = state_dot_func(t, x, u_val, m_val, M_val, L_val, g)
    return np.array(xdot, dtype=float)

# Simulation parameters
m_val = 10.0
M_val = 20.0
L_val = 2.0
g = 9.81

x0 = np.array([0.0, 1.0, 0.0, 0.0])  # initial state [q, qd]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 200)

# Run simulation
solution = solve_ivp(system, t_span, x0, t_eval=t_eval)

# Extract results
t_vals = solution.t
y_vals = solution.y[0]    # position
ydot_vals = solution.y[1] # velocity
F_vals = np.array([u_func(ti, [yi, vi]) for yi, vi, ti in zip(y_vals, ydot_vals, t_vals)])

# --- Generate plots of the cart simulation! ---
plot_response_cartpole(solution, F_vals, t_vals, file_directory, filename="response_plot.png")

# --- Animation ---
cart_anim_file = str(file_directory)+"/Cart_simulation.gif"
animate_cartpole(solution.y.T, F_sol=F_vals, t_vec=t_vals, file_name=cart_anim_file,L=L_val)