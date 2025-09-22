import numpy as np
import control as ct
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PendulumPlotting import rot_pendulum_animator, plot_response
from DynamicsFuncs import rot_pend_dynamics_num
#Import the dynamic model functions
from PendulumModelling import M_f, N_f
file_directory = os.path.dirname(os.path.abspath(__file__))

equilibrium = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)  # [θ1, θ2, θ1d, θ2d]

A = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 57.144539, 0.0, 1.2051996]
])

B = [[0.], [0.], [ 1.], [0.83706176]]

C = np.eye(4)  # Identity output matrix (just for completeness)

D = np.zeros((4, 1))  # No direct feedthrough

sys_c = ct.ss(A, B, C, D)
sys_tf = ct.ss2tf(sys_c)
print("All transfer functions:",sys_tf)

G = sys_tf[1,0]
print("Input-to-arm TF:",G)
print("Poles:", ct.poles(G))

alpha = 100  # fixed zero location
Kd = 1      # gain to sweep
s = ct.tf('s')
C = Kd * (s + alpha)  # initial value
L=C*G

K=1.0
T = ct.feedback(G, 1)   # unity feedback

ct.root_locus(L, gains=np.linspace(0, 1000, 500))

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

kd = 242
kp = alpha * kd
# Control input function
def u(t,x):
    #acceleration = kp*(equilibrium[1]-x[1])

    acceleration = kp*(equilibrium[1]-x[1]) -kd*x[3]

    return np.array([acceleration]).flatten()

# ---------------------------------------------------------------------
# 3) Simulate with SciPy
# ---------------------------------------------------------------------

# Dynamics function for ODE solver
def xdot(t, x):
    control = u(t,x)
    return rot_pend_dynamics_num(x, control, M_f, N_f)

t_span = (0.0, 5.0)
t_eval = np.linspace(t_span[0], t_span[1], 400)
x0 = np.array([0.0, np.pi-0.01, 0.0, 0.0], dtype=float)

sol = solve_ivp(xdot, t_span, x0, t_eval=t_eval, method="RK45", atol=1e-9, rtol=1e-7)

F = np.array([u(tt, xx) for tt, xx in zip(sol.t, sol.y.T)])

# --- Plot results ---
plot_response(sol, F, file_directory, filename="response_plot.png")

# --- Animate results ---
rot_pendulum_animator(sol, name=os.path.join(file_directory, "rotary_pendulum_anim"))
