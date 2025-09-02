import numpy as np
import control

A = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 57.144539, 0.0, 1.2051996]
])

B = [[0.], [0.], [ 1.], [0.83706176]]

C = np.eye(4)  # Identity output matrix (just for completeness)
D = np.zeros((4, 1))  # No direct feedthrough

sys_c = control.ss(A, B, C, D)

# Set your sampling time (e.g., 0.01 seconds)
Ts = 0.01

# Convert to discrete time
sys_d = control.c2d(sys_c, Ts, method='zoh')  # Zero-order hold

# Extract discrete A, B, C, D matrices
Ad, Bd, Cd, Dd = control.ssdata(sys_d)
print(sys_d)
# Desired poles
desired_poles =[-5.69167035, -8.2767094, -2.843+1.66j, -2.843-1.66j]

# Pole placement
place_obj_c = control.place(A, B, desired_poles)
print(place_obj_c)

# Pole placement
place_obj_d = control.place(Ad, Bd, desired_poles)

print(place_obj_d)

Q=np.diag([0.1,0.01,0.01,0.01]) 
R=0.001

K, S, E = control.dlqr(Ad, Bd, Q, R)

# Closed-loop system matrix
A_cl = A - B @ K

# Closed-loop poles
poles = np.linalg.eigvals(A_cl)

print("LQR Gain K:\n", K)
print("Closed-loop poles:\n", poles)