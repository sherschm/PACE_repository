from sympy import symbols, Function, diff, Eq, simplify, pprint

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
# Step 2: Define kinetic energy only (no potential energy)
# -----------------------------
print("Step 2: Defining kinetic energy...")
T = (1/2) * m * diff(y, t)**2  # kinetic energy
print("Kinetic Energy T =")
pprint(T)
print("\n")

# -----------------------------
# Step 3: Construct the Lagrangian
# -----------------------------
print("Step 3: Constructing the Lagrangian (no potential energy)...")
L = T
print("Lagrangian L =")
pprint(L)
print("\n")

# -----------------------------
# Step 4: Apply Lagrange-d'Alembert Equation with Q term
# -----------------------------
print("Step 4: Applying Lagrange-d'Alembert equation with generalized force Q...")
y_dot = diff(y, t)
dL_dy = diff(L, y)               # ∂L/∂y
dL_dy_dot = diff(L, y_dot)       # ∂L/∂y_dot
d_dt_dL_dy_dot = diff(dL_dy_dot, t)  # d/dt(∂L/∂y_dot)

Q = u  # generalized force
EL_eq = Eq(d_dt_dL_dy_dot - dL_dy, Q)
print("Lagrange-d'Alembert equation:")
pprint(EL_eq)
print("\n")

# -----------------------------
# Step 5: Simplify to Newton's Law
# -----------------------------
print("Step 5: Simplifying equation to F = m a...")
a = diff(y, t, t)  # acceleration
EL_eq_simplified = simplify(EL_eq.subs(diff(y, t, t), a))
print("Final simplified equation:")
pprint(EL_eq_simplified)
print("This is Newton's second law derived using Lagrange-d'Alembert with a generalized force.")
