import sympy as sp
from sympy import symbols, Function, diff, Eq, simplify, pprint
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import animate_cart, plot_response
import os
file_directory = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Step 1: Define symbols
# -----------------------------
print("Step 1: Defining variables...")
t = symbols('t')  # time
m, u = symbols('m u')  # mass and applied force
y = Function('y')(t)  # position as a function of time
print("Time variable:")
pprint(t)
print("Mass and Applied Force:")
pprint((m, u))
print("Position:")
pprint(y)
print("\n")

# -----------------------------
# Step 2: Define energy terms
# -----------------------------
print("Step 2: Defining energy terms...")
T = (1/2) * m * diff(y, t)**2  # kinetic energy
print("Kinetic Energy T =")
pprint(T)
print("\n")

V = 0  # potential energy (zero for this example)
print("Potential Energy V =")
pprint(V)
print("\n")

# -----------------------------
# Step 3: Construct the Lagrangian
# -----------------------------
print("Step 3: Constructing the Lagrangian (no potential energy)...")
L = T - V
print("Lagrangian L =")
pprint(L)
print("\n")

# -----------------------------
# Step 4: Apply Lagrange-d'Alembert Equation with Q term
# -----------------------------
print("Step 4: Applying Lagrange-d'Alembert equation with generalized force Q...")
y_dot = diff(y, t)
y_ddot = diff(y, t, t)
dL_dy = diff(L, y)               # ∂L/∂y
dL_dy_dot = diff(L, y_dot)       # ∂L/∂y_dot
d_dt_dL_dy_dot = diff(dL_dy_dot, t)  # d/dt(∂L/∂y_dot)

Q = u  # generalized force
EL_eq = Eq(d_dt_dL_dy_dot - dL_dy, Q)
print("Lagrange-d'Alembert equation:")
pprint(EL_eq)
print("\n")
print("This is Newton's second law rearranged!")

# -----------------------------
# Step 6: Simulate the system!
# -----------------------------

# Symbolic dynamics
x1 = y                   # position
x2 = y_dot               # velocity
x1_dot = x2
x2_dot = sp.solve(EL_eq, y_ddot)[0]   # solve for acceleration

# State vector and dynamics
state = sp.Matrix([x1, x2])
state_dot = sp.Matrix([x1_dot, x2_dot])

# Convert symbolic model to numerical function
state_dot_func = sp.lambdify((t, state, u, m), list(state_dot), "numpy")

# Control input (scalar force)
def u_func(t, state):
    return 1.0  # constant input

# System dynamics for solver
def system(t, x):
    u_val = float(u_func(t, x))
    return np.array(state_dot_func(t, x, u_val, m_val), dtype=float)

# Simulation parameters
m_val = 10.0
x0 = np.array([0.0, 0.0])  # initial [position, velocity]
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
plot_response(y_vals, ydot_vals, F_vals, t_vals, file_directory, filename="response_plot.png")

# --- Animation ---
cart_anim_file = str(file_directory)+"/Cart_simulation.gif"
animate_cart(x_sol=y_vals, F_sol=F_vals, t_vec=t_vals, file_name=cart_anim_file)