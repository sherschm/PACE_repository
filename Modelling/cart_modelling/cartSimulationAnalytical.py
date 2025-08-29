import numpy as np
import matplotlib.pyplot as plt
import os
from cartPlotting import animate_cart
file_directory = os.path.dirname(os.path.abspath(__file__))

#System parameters
m = 10  # kg
F = 1 #Newton

#Simulation parameters
t_step = 0.05 #seconds 
t_final = 10.0 #seconds 
t = np.arange(0.0,t_final,t_step) #define a series of time steps
n = t.shape[0] #number of time steps in simulation

#Initial conditions
y0 = 0.0 #initial position
yd0 = 0.0 #initial velocity

#Analytical simulation solution
y= (F/(2*m))*t**2 + yd0*t + y0

# --- Plot results ---
plt.figure()
plt.plot(list(t), list(y))
plt.xlabel("time (s)")
plt.ylabel("y (m)")
plt.grid(True)
plt.savefig(str(file_directory)+"/response_plot.png", dpi=150)

plt.figure()
plt.plot([0.0,t_final], [F,F])
plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.savefig(str(file_directory)+"/force_plot.png", dpi=150)

# --- Generate visualisation of the cart simulation! ---
animate_cart(x_sol=list(y), F_sol=[F]*n, t_vec=list(t), file_name=str(file_directory)+"/Cart_simulation.gif")