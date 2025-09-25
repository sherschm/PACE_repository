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

M = np.array([[m, 0.0],
              [0.0, m]])

P = np.array([[k, -k],
              [-k, k]])

Damp = np.array([[c, -c],
                 [-c, c]])

b = np.array([[1.0],
              [0.0]])

# --- State-space matrices ---
A = np.block([
    [np.zeros((2,2)), np.identity(2)],
    [inv(M) @ -P, -Damp]
])

B = np.block([
    [np.zeros((2,1))],
    [inv(M) @ b]
])

# Step 1: Eigenvalues of A
eigvals, eigvecs = np.linalg.eig(A)

# Step 2: Natural frequencies = magnitude of eigenvalues
natural_freqs = np.abs(eigvals)

C = np.identity(4)  # we measure cart 3 position

D = np.zeros((4,1))

#Build state space model object
sys_ss = ct.ss(A, B, C, D)

# --- Convert to transfer function ---
sys_tf = ct.ss2tf(sys_ss)
print("Transfer function:",sys_tf)

G_y2 = sys_tf[1,0]
G_y2 = clean_transfer_function(G_y2) # remove nasty small terms

print("Transfer function:",G_y2)

# --- Poles, zeros ---
print("Poles:", ct.poles(G_y2))
print("Zeros:", ct.zeros(G_y2))

ct.pole_zero_plot(G_y2)
plt.savefig(os.path.join(plot_dir, "PoleZero.png")) # Save the figure in the subfolder

ct.root_locus(G_y2) 
plt.savefig(os.path.join(plot_dir, "RootLocusP.png")) # Save the figure in the subfolder
#plt.show()

alpha =1.0  # fixed zero location
Kd = 1      # gain to sweep
s = ct.tf('s')
K = Kd * (s + alpha)  # initial value
L=K*G_y2

# --- Closed-Loop Poles, zeros ---
print("Closed-loop Poles:", ct.poles(K*G_y2))
print("Closed-loop Zeros:", ct.zeros(K*G_y2))

# --- Control parameters ---
ydot_d = 0.0  # desired velocity
k_d = 4.61
k_p= k_d*alpha  # PD gains
#plot PD Root Locus
K = k_d * (s + alpha)  # initial value
L=K*G_y2

#Plot Open-loop Bode plot
ct.bode_plot(G_y2, dB=True, Hz=False, omega_limits=(1e-2, 1e2), omega_num=400,label="Open-loop")
#Plot Closed-loop Bode plot
ct.bode_plot(L, dB=True, Hz=False, omega_limits=(1e-2, 1e2), omega_num=400,label="Closed-loop")
plt.savefig(os.path.join(plot_dir, "Bode.png")) # Save the figure in the subfolder

ct.root_locus(L) 
plt.savefig(os.path.join(plot_dir, "RootLocusPD.png")) # Save the figure in the subfolder
#plt.show()

# margins
gm, pm, wg, wp = ct.margin(L)
print("Gain margin:", gm, "at freq", wg)
print("Phase margin:", pm, "at freq", wp)

def u(t, x):
    y2 = x[1] #cart position
    y2_dot = x[3] #cart velocity

    # Control Law (tune this!)
    force = -k_p * y2 - k_d * y2_dot #this is equivalent to 
    return [force]


# Dynamics function
def xdot(t, x):
    dx = A @ x + B @ u(t, x)
    return dx.flatten()
    
# --- Simulation setup ---
t_span = (0, 10)  # time range
x0 = [0.0, 1.0, 0.5, 0.5]  # initial state: [x1, x2, x1dot, x2dot]

solution = solve_ivp(xdot, t_span, x0, t_eval=np.linspace(*t_span, 400))

# Extract simulation data
t = solution.t
y = solution.y[0:2, :]    # positions of the 3 carts
ydot = solution.y[2:4, :] # velocities of the 3 carts

# Force history (only applied to cart 1)
F = np.array([u(ti, xi)[0] for ti, xi in zip(t, solution.y.T)])

# --- Generate plots of the cart simulation - CART 2 ONLY! ---
plot_response( solution.y[1, :], solution.y[3, :], F, t, plot_dir, filename="response_plot.png")

# --- Generate visualisation of the cart simulation! ---
animate_cart(x_sol=y, F_sol=F, t_vec=t, file_name=str(plot_dir)+"/Cart_simulation.gif")

def control_energy(force, velocity, time):
    #Compute the total energy used by the controller throughout the simulation.
    power = np.abs(force * velocity)   # instantaneous power (W = J/s)
    energy = np.trapezoid(power, time)  # integrate power over time
    return energy

energy_used = control_energy(F, solution.y[2, :] , t)

print(f"Total energy used by the controller: {energy_used:.2f} Joules")
plt.close()
