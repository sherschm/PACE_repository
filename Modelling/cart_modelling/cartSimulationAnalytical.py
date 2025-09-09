#import some useful Python libraries
import numpy as np #This is a package for doing maths
import os #Package for working with directories
from cartPlotting import animate_cart, plot_response #import some plotting functions from another script

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
y= (F/(2*m))*t**2 + yd0*t + y0 # "**2" means to the power of 2
ydot= (F/m)*t + yd0

# --- Plot results ---
plot_response(y, ydot, [F]*n, t, file_directory, filename="response_plot.png")

# --- Generate visualisation of the cart simulation! ---
animate_cart(x_sol=list(y), F_sol=[F]*n, t_vec=list(t), file_name=str(file_directory)+"/Cart_simulation.gif")