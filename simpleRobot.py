import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L1 = 1.0  # Length of link 1
L2 = 1.0  # Length of link 2
K = 1.0   # Proportional gain for the P controller
dt = 0.01 # Time step for simulation
total_time = 5.0  # Total simulation time
steps = int(total_time / dt)

# Desired angles (in radians)
theta1_des = np.pi / 6  
theta2_des = -np.pi / 2 

# Initial angles
theta1 = 0.0
theta2 = 0.0

# Arrays to store the trajectory
theta1_traj = np.zeros(steps)
theta2_traj = np.zeros(steps)
x1_traj = np.zeros(steps)
y1_traj = np.zeros(steps)
x2_traj = np.zeros(steps)
y2_traj = np.zeros(steps)

# Simulate the motion
for i in range(steps):
    # Compute errors
    error1 = theta1_des - theta1
    error2 = theta2_des - theta2
    
    # P controller: set velocities proportional to errors
    dtheta1 = K * error1
    dtheta2 = K * error2
    
    # Update angles
    theta1 += dtheta1 * dt
    theta2 += dtheta2 * dt
    
    # Compute positions for plotting
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    # Store for animation
    theta1_traj[i] = theta1
    theta2_traj[i] = theta2
    x1_traj[i] = x1
    y1_traj[i] = y1
    x2_traj[i] = x2
    y2_traj[i] = y2

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)

# Lines for the robot links
line, = ax.plot([], [], 'o-', lw=2)
target_point = ax.plot(L1 * np.cos(theta1_des) + L2 * np.cos(theta1_des + theta2_des),
                       L1 * np.sin(theta1_des) + L2 * np.sin(theta1_des + theta2_des),
                       'rx', markersize=10)[0]

# Animation function
def animate(i):
    # Update the line data for the current frame
    line.set_data([0, x1_traj[i], x2_traj[i]], [0, y1_traj[i], y2_traj[i]])
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=steps, interval=dt*1000, blit=True)

# Display the animation
plt.title('2-Link Robot Arm with P Controller')
plt.show()