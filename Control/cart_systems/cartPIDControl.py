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

d = 20 # disturbance force

# Desired motion
y_d = 2.0      # desired position (y_d)
ydot_d = 0.0  # desired velocity

# PID gains
k_p, k_i, k_d = 25.0, 5.0, 10.0

# PID control law
def u(t,x_e):

    y = x_e[0]
    ydot = x_e[1]
    e_int = x_e[2]

    # Error
    e = y_d - y

    force = k_p * e + k_d * (ydot_d - ydot) + k_i * e_int
    return np.array([force])


# --- Dynamics with PID ---
def xe_dot(t, x_e):
    """
    x_e = [y, ydot, e_int]
    """
    y, ydot, e_int = x_e

    # Error
    e = y_d - y

    # Cart dynamics: dx = A*[y, ydot] + B * u_{PID} + disturbance
    state = np.array([y, ydot])
    dx = A @ state + B.flatten() * (u(t,x_e)) - np.array([0, d/m])

    # Error integral dynamics
    de_int = e

    return np.hstack([dx, de_int])

# --- Initial conditions & simulation parameters ---
t_span = (0, 10)
x0 = np.array([0.0, 0.0, 0.0])   # [y, ydot, e_int]

# Solve the ODE
solution = solve_ivp(xe_dot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation data
t = solution.t
y = solution.y[0]
ydot = solution.y[1]
e_int = solution.y[2]
F = np.array([u(ti,[yi, ydoti, e_inti])[0] for yi, ydoti, e_inti, ti in zip(y, ydot, e_int, t)])

# --- Generate plots of the cart simulation! ---
plot_response( y, ydot, F, t, file_directory, filename="response_plot.png")

# --- Generate visualisation of the cart simulation! ---
animate_cart(x_sol=np.array([y]), F_sol=F, t_vec=t, file_name=str(file_directory)+"/Cart_simulation.gif")

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