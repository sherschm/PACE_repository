import numpy as np
import control as ct

A = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 57.144539, 0.0, 1.2051996]
])

B = [[0.], [0.], [ 1.], [0.83706176]]

C = np.eye(4)  # Identity output matrix (just for completeness)
#C = [[0.], [1.], [ 0.], [0.]]
D = np.zeros((4, 1))  # No direct feedthrough

sys_c = ct.ss(A, B, C, D)
sys_tf = ct.ss2tf(sys_c)
print("All transfer functions:",sys_tf)

G = sys_tf[1,0]
print("Input-to-arm TF:",G)
print("Poles:", ct.poles(G))

K=1.0
T = ct.feedback(K*G, 1)   # unity feedback

ct.root_locus(T, gains=np.linspace(0, 1000, 500))

# Set your sampling time (e.g., 0.01 seconds)
Ts = 0.01

# Convert to discrete time
sys_d = ct.c2d(sys_c, Ts, method='zoh')  # Zero-order hold

# Extract discrete A, B, C, D matrices
Ad, Bd, Cd, Dd = ct.ssdata(sys_d)
print(sys_d)
# Desired poles
desired_poles =[-5.69167035, -8.2767094, -2.843+1.66j, -2.843-1.66j]

# Pole placement
place_obj_c = ct.place(A, B, desired_poles)
print(place_obj_c)

# Pole placement
place_obj_d = ct.place(Ad, Bd, desired_poles)

print(place_obj_d)

Q=np.diag([0.1,0.01,0.01,0.01]) 
R=0.001

K, S, E = ct.dlqr(Ad, Bd, Q, R)

# Closed-loop system matrix
A_cl = A - B @ K

# Closed-loop poles
poles = np.linalg.eigvals(A_cl)

print("LQR Gain K:\n", K)
print("Closed-loop poles:\n", poles)