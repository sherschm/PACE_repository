import numpy as np

# Constants
pi = np.pi

# L-bar dimensions and material
r = 0.004     # Outer radius (m)
ri = 0.0032   # Inner radius (m)

# Geometry
l1 = 0.125  # Horizontal section (m)
l2 = 0.25   # Vertical section (m)

# Masses
m_bar = 0.0174    # Total measured bar mass (kg)
rotor_mass = 0.031 # encoder rotor mass (guess, based on specs)

m1 = m_bar * l1 / (l1 + l2)
m2 = m_bar * l2 / (l1 + l2)
m_tip = 0.00 #extra mass on tip
m = m1 + m2 + m_tip + rotor_mass #total mass
g = 9.81

# Center of mass calculation
rc = (1/m) * (m1 * np.array([0, l1/2, 0]) +
              m2 * np.array([0, l1, -l2/2]) +
              m_tip * np.array([0, l1, -l2]))

# Damping matrix (scalar or matrix form)
damping = 0.0003  # or np.array([[0, 0], [0, 0.0002]])

# Inertia tensor function
def I_tube(r1, r2, h, mi, offset, axis):
    r_avg2 = (r1**2 + r2**2) / 2

    if axis == "x":
        Ix = 0.5 * mi * r_avg2
        Iy = (1/12) * mi * (3*r_avg2 + h**2)
        Iz = Iy
    elif axis == "y":
        Iy = 0.5 * mi * r_avg2
        Ix = (1/12) * mi * (3*r_avg2 + h**2)
        Iz = Ix
    elif axis == "z":
        Iz = 0.5 * mi * r_avg2
        Ix = (1/12) * mi * (3*r_avg2 + h**2)
        Iy = Ix
    else:
        raise ValueError("Invalid axis")

    I_diag = np.diag([Ix, Iy, Iz])
    offset = np.array(offset).reshape(3, 1)
    parallel_axis_term = mi * ((np.dot(offset.T, offset) * np.eye(3)) - np.dot(offset, offset.T))
    I_out = I_diag + parallel_axis_term

    return I_out

# Inertia tensors
I1 = I_tube(r, ri, l1, m1, [0, l1/2, 0], "y")
I2 = I_tube(r, ri, l2, m2, [0, l1, -l2/2], "z")
Irotor = np.diag([0, 3.5e-6, 0]) #guess
I3 = m_tip * np.diag([0, l1**2, l2**2])

# Combine inertias and apply correction factor
Ip = 0.85 * (I1 + I2 + I3)  # If including rotor: + Irotor

# Frequency (optional)
observed_freq = 9 / 7.4625

# Final dynamic parameters tuple
dyn_params = (rc, m, Ip, g, damping)

# Encoder steps per radian
encoder_steps_per_rad = pi / 1200
