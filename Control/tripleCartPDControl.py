import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import animate_cart, plot_response
import control as ct
import os
file_directory = os.path.dirname(os.path.abspath(__file__))

# --- System Parameters ---
m = 10  # kg (cart mass)
c = 0.0  # Ns/m (viscous friction coefficient)
k = 10 #N/m

M = np.array([[m, 0, 0],
              [0, m, 0],
              [0, 0, m]])

P = np.array([[k, -k,    0],
              [-k, k+k, -k],
              [0, -k,    k]])

b = np.array([[1],
              [0],
              [0]])

# --- State-space matrices ---
A = np.block([
    [np.zeros((3,3)), np.identity(3)],
    [inv(M) @ -P, np.zeros((3,3))]
])

B = np.block([
    [np.zeros((3,1))],
    [inv(M) @ b]
])

C = np.array([[0., 0., 1., 0., 0., 0.]])  # we measure cart 3 position

D = np.array([[0.0]])

sys_ss = ct.ss(A, B, C, D)

# --- Convert to transfer function ---
sys_tf = ct.ss2tf(sys_ss)
print("Transfer function:",sys_tf)

# --- Poles, zeros ---
print("Poles:", ct.poles(sys_ss))
print("Zeros:", ct.zeros(sys_ss))

ct.bode_plot(sys_tf, dB=True, Hz=False, omega_limits=(1e-1, 1e2), omega_num=400)

# --- Control parameters ---
y_d = 2.0     # desired position for cart 1
ydot_d = 0.0  # desired velocity
k_p, k_d = 10.0, 0.0  # PD gains

# Control input function (force applied to cart 1)
def u(t, x):
    y = x[2]       # cart 1 position
    ydot = x[5]    # cart 1 velocity
    force = k_p * (y_d - y) + k_d * (ydot_d - ydot)
    return np.array([force])

desired_poles =[-0.1, -0.2, -0.8+1.66j, -0.8-1.66j, -1+1.66j, -1-1.66j]
K = ct.place(A, B, desired_poles)

def u(t, x):
    # full-state feedback, returns 1D array
    force = -K @ (x)  
    return np.array([force])[0]

#def u(t, x):
#    y = x[0]       # cart 1 position
#    ydot = x[3]    # cart 1 velocity
#    force = 0.0
#    return np.array([force])

# Dynamics function
def xdot(t, x):
    dx = A @ x + B @ u(t, x)
    return dx.flatten()
    
# --- Simulation setup ---
t_span = (0, 10)  # time range
x0 = np.ones(6)  # initial state: [x1, x2, x3, x1dot, x2dot, x3dot]

solution = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation data
t = solution.t
y = solution.y[0:3, :]    # positions of the 3 carts
ydot = solution.y[3:6, :] # velocities of the 3 carts

# Force history (only applied to cart 1)
F = np.array([u(ti, xi)[0] for ti, xi in zip(t, solution.y.T)])
# --- Generate plots of the cart simulation! ---
plot_response( solution.y[3, :], solution.y[5, :], F, t, file_directory, filename="response_plot.png")

# --- Generate visualisation of the cart simulation! ---
animate_cart(x_sol=y, F_sol=F, t_vec=t, file_name=str(file_directory)+"/Cart_simulation.gif")

def control_energy(force, velocity, time):
    """
    Compute the total energy used by the controller throughout the simulation.
    
    Parameters
    ----------
    force : array_like
        Force applied by the controller at each timestep (N).
    velocity : array_like
        Cart velocity at each timestep (m/s).
    time : array_like
        Time vector corresponding to force/velocity samples (s).
        
    Returns
    -------
    energy : float
        Total energy used (Joules).
    """
    power = np.abs(force * velocity)   # instantaneous power (W = J/s)
    energy = np.trapezoid(power, time)  # integrate power over time
    return energy

energy_used = control_energy(F, solution.y[3, :] , t)

print(f"Total energy used by the controller: {energy_used:.2f} Joules")