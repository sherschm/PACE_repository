import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cartPlotting import animate_cart, plot_response
import os
file_directory = os.path.dirname(os.path.abspath(__file__))

# --- System Parameters ---
m = 10  # kg (cart mass)
c = 0.0  # Ns/m (viscous friction coefficient)

# State-space representation of the cart system dynamics
A = np.array([[0, 1],
              [0, -c/m]])
B = np.array([[0],
              [1/m]])

# Desired motion
y_d = 2.0      # desired position (y_d)
ydot_d = 0.0  # desired velocity

# PD gains
k_p, k_d = 50, 50.0

# Control input function
def u(t,x):
    y = x[0]
    ydot = x[1]
    force = k_p * (y_d - y) + k_d * (ydot_d - ydot) # PD control law
    return np.array([force])

# Dynamics function
def xdot(t, x):
    dx = A @ x + B @ u(t,x)
    return dx.flatten()

# Initial conditions & simulation parameters
t_span = (0, 10)   # time range
x0 = np.array([0.0, 0.0])    # initial state: [x, xdot]

# Solve the ODE
solution = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation data
t = solution.t
y = solution.y[0]
ydot = solution.y[1]
F = np.array([u(ti,[yi, ydoti])[0] for yi, ydoti, ti in zip(y, ydot, t)])

# --- Generate plots of the cart simulation! ---
plot_response( y, ydot, F, t, file_directory, filename="response_plot.png")

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

energy_used = control_energy(F, ydot, t)

print(f"Total energy used by the controller: {energy_used:.2f} Joules")