# Python Tutorial: Deriving F=ma Using the Euler-Lagrange Equation with SymPy

from sympy import symbols, Function, diff, Eq, simplify

# Define symbols
t = symbols('t')
m, F = symbols('m F')
x = Function('x')(t)

# Kinetic energy: T = (1/2) m v^2
T = (1/2) * m * diff(x, t)**2

# Potential energy: V = -F x
V = -F * x

# Lagrangian: L = T - V
L = T - V

# Euler-Lagrange equation: d/dt(∂L/∂x_dot) - ∂L/∂x = 0
x_dot = diff(x, t)
dL_dx = diff(L, x)
dL_dx_dot = diff(L, x_dot)
d_dt_dL_dx_dot = diff(dL_dx_dot, t)

# Euler-Lagrange equation
EL_eq = Eq(d_dt_dL_dx_dot - dL_dx, 0)
print("Euler-Lagrange equation:")
print(EL_eq)

# Simplify to get F = m a
# Substitute x'' for acceleration
a = diff(x, t, t)
EL_eq_simplified = simplify(EL_eq.subs(diff(x, t, t), a))
print("\nSimplified equation (F = m a):")
print(EL_eq_simplified)
