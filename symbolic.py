from sympy import symbols, cos, sin, diff, Matrix, simplify, Rational, pprint, expand
from sympy import pretty

# Define symbolic variables
theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2 = symbols('theta1 theta2 dtheta1 dtheta2 ddtheta1 ddtheta2')
m1, m2, l1, l2, g = symbols('m1 m2 l1 l2 g')
I1, I2 = symbols('I1 I2')  # Moments of inertia about the center of mass of each link

# Positions of the centers of mass (at the midpoint of each link)
x1 = (l1 / 2) * cos(theta1)  # Center of mass of link 1
y1 = (l1 / 2) * sin(theta1)
x2 = l1 * cos(theta1) + (l2 / 2) * cos(theta1 + theta2)  # Center of mass of link 2
y2 = l1 * sin(theta1) + (l2 / 2) * sin(theta1 + theta2)

# Velocities of the centers of mass
dx1 = -(l1 / 2) * sin(theta1) * dtheta1
dy1 = (l1 / 2) * cos(theta1) * dtheta1
dx2 = -l1 * sin(theta1) * dtheta1 - (l2 / 2) * sin(theta1 + theta2) * (dtheta1 + dtheta2)
dy2 = l1 * cos(theta1) * dtheta1 + (l2 / 2) * cos(theta1 + theta2) * (dtheta1 + dtheta2)

# Kinetic energy: translational + rotational
# Translational: (1/2) m (dx^2 + dy^2)
# Rotational: (1/2) I w^2, where w is the angular velocity (dtheta1 for link 1, dtheta1 + dtheta2 for link 2)
T1 = Rational(1, 2) * m1 * (dx1**2 + dy1**2) + Rational(1, 2) * I1 * dtheta1**2
T2 = Rational(1, 2) * m2 * (dx2**2 + dy2**2) + Rational(1, 2) * I2 * (dtheta1 + dtheta2)**2
T = T1 + T2
T = simplify(T)

# Potential energy (center of mass at midpoints)
V = g * (m1 * y1 + m2 * y2)

# Lagrangian
L = T - V

# Compute tau1
dL_d_dtheta1 = diff(L, dtheta1)
ddt_dL_d_dtheta1 = diff(dL_d_dtheta1, theta1) * dtheta1 + diff(dL_d_dtheta1, theta2) * dtheta2 + \
                   diff(dL_d_dtheta1, dtheta1) * ddtheta1 + diff(dL_d_dtheta1, dtheta2) * ddtheta2
dL_d_theta1 = diff(L, theta1)
tau1 = expand(ddt_dL_d_dtheta1 - dL_d_theta1)

# Compute tau2
dL_d_dtheta2 = diff(L, dtheta2)
ddt_dL_d_dtheta2 = diff(dL_d_dtheta2, theta1) * dtheta1 + diff(dL_d_dtheta2, theta2) * dtheta2 + \
                   diff(dL_d_dtheta2, dtheta1) * ddtheta1 + diff(dL_d_dtheta2, dtheta2) * ddtheta2
dL_d_theta2 = diff(L, theta2)
tau2 = expand(ddt_dL_d_dtheta2 - dL_d_theta2)

# Extract mass matrix M
M11 = simplify(tau1.coeff(ddtheta1))
M12 = simplify(tau1.coeff(ddtheta2))
M21 = simplify(tau2.coeff(ddtheta1))
M22 = simplify(tau2.coeff(ddtheta2))
# Create the mass matrix with extra spacing between elements for clearer output

M = Matrix([[M11, M12], [M21, M22]])



# Coriolis + gravity terms
coriolis_grav1 = simplify(tau1 - M11 * ddtheta1 - M12 * ddtheta2)
coriolis_grav2 = simplify(tau2 - M21 * ddtheta1 - M22 * ddtheta2)

# Gravity vector G
G1 = coriolis_grav1.subs({dtheta1: 0, dtheta2: 0})
G2 = coriolis_grav2.subs({dtheta1: 0, dtheta2: 0})
G = Matrix([G1, G2])

# Coriolis and centrifugal vector
C_vec1 = simplify(coriolis_grav1 - G1)
C_vec2 = simplify(coriolis_grav2 - G2)
C_vec = Matrix([C_vec1, C_vec2])


print('Mass matrix M:')
pprint(M)
print('\nGravity vector G:')
pprint(G)
print('\nCoriolis and centrifugal vector:')
pprint(C_vec)