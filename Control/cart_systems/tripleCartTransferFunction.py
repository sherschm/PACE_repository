import matplotlib.pyplot as plt
plt.close('all')   # closes all existing figure windows
import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from cartPlotting import animate_cart, plot_response
from cartHelperFuncs import clean_transfer_function
import control as ct  
import os
file_directory = os.path.dirname(os.path.abspath(__file__))
plot_dir = str(file_directory)+"/plots/"

# --- System Parameters ---
m = 10.0  # kg (cart mass)
c = 0.3  # Ns/m (viscous friction coefficient)
k = 10.0 #N/m

M = np.array([[m, 0.0, 0.0],
              [0.0, m, 0.0],
              [0.0, 0.0, m]])

P = np.array([[k, -k,    0.0],
              [-k, k+k, -k],
              [0.0, -k,    k]])

Damp = np.array([[c, -c,    0.0],
              [-c, c+c, -c],
              [0.0, -c,    c]])

b = np.array([[1.0],
              [0.0],
              [0.0]])

# --- State-space matrices ---
A = np.block([
    [np.zeros((3,3)), np.identity(3)],
    [inv(M) @ -P, -Damp]
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

G = sys_tf[0,0]
G = clean_transfer_function(G) # remove nasty small terms

print("Transfer function:",G)

# --- Poles, zeros ---
print("Poles:", ct.poles(G))
print("Zeros:", ct.zeros(G))

#T = ct.feedback(G, 1)   # unity feedback

alpha = 1  # fixed zero location
Kd = 1      # gain to sweep
s = ct.tf('s')
C = Kd * (s + alpha)  # initial value
L=C*G

# --- Poles, zeros ---
print("Poles:", ct.poles(L))
print("Zeros:", ct.zeros(L))

ct.root_locus(G, gains=np.linspace(0, 10000, 500)) 
plt.savefig(os.path.join(plot_dir, "RootLocusP.png")) # Save the figure in the subfolder

# --- Control parameters ---
ydot_d = 0.0  # desired velocity
k_d = 24.0
k_p= k_d*alpha  # PD gains

ct.bode_plot(sys_tf, dB=True, Hz=False, omega_limits=(1e-2, 1e2), omega_num=400)
plt.savefig(os.path.join(plot_dir, "Bode.png")) # Save the figure in the subfolder

desired_poles =[-0.1, -0.2, -0.8+1.66j, -0.8-1.66j, -1+1.66j, -1-1.66j]
K = ct.place(A, B, desired_poles)

def u(t, x):
    # LQR full-state feedback, returns 1D array
    force = -k_p * x[2] -k_d * x[5] # regulation (get to zero)
    return [force]

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
t_span = (0, 20)  # time range
x0 = np.ones(6)  # initial state: [x1, x2, x3, x1dot, x2dot, x3dot]

solution = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation data
t = solution.t
y = solution.y[0:3, :]    # positions of the 3 carts
ydot = solution.y[3:6, :] # velocities of the 3 carts

# Force history (only applied to cart 1)
F = np.array([u(ti, xi)[0] for ti, xi in zip(t, solution.y.T)])

# --- Generate plots of the cart simulation - CART 3 ONLY! ---
plot_response( solution.y[2, :], solution.y[5, :], F, t, plot_dir, filename="response_plot_Cart3.png")

# --- Generate visualisation of the cart simulation! ---
animate_cart(x_sol=y, F_sol=F, t_vec=t, file_name=str(plot_dir)+"/Cart_simulation.gif")

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