import sympy as sp
from sympy import symbols, Function, diff, Eq, simplify, pprint
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import animate_cart
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
x_dot1 = y_dot                  # dx/dt = v
x_dot2 = sp.solve(EL_eq, y_ddot)[0]   # -> rearrange the symbolic equation for y_ddot

x = sp.Matrix([y, y_dot])  # state vector
x_dot = sp.Matrix([x_dot1, x_dot2])  # first-order state-space model

x_dot_func = sp.lambdify((t, x, u, m), list(x_dot), 'numpy') # convert model to a numerical function

# Define control input
def u_func(t,x):
    force = 1.0  # Constant control input for this example
    return np.float64(force)

def system(t, x):
    u_val = u_func(t,x)
    return np.array(x_dot_func(t, x, u_val, m_val)).flatten()

m_val = 10.0
x0 = np.array([0.0, 0.0])  # initial position and velocity
t_span = (0, 10)
t_eval = np.linspace(*t_span, 200)

solution = solve_ivp(system, t_span, x0, t_eval=t_eval)

# Extract simulation results and visualise
t = solution.t
x = solution.y[0]
xdot = solution.y[1]
F = np.array([u_func(ti,[xi, vi]) for xi, vi, ti in zip(x, xdot, t)])

# --- Plot results ---
plt.figure()
plt.plot(t, x, label="y (m)")
plt.plot(t, xdot, label="dy/dt (m/s)")
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
cart_anim_file = str(file_directory)+"/Cart_simulation.gif"
animate_cart(x_sol=x, F_sol=F, t_vec=t, file_name=cart_anim_file)