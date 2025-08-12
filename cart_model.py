import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- System Parameters ---
m = 10  # kg
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])

# Desired motion
cmnd = 7.0      # desired position
cmnd_dot = 0.0  # desired velocity

# PD gains
K_p, K_d = 100, 60

# Control input function
def u(q, t):
    return K_p * (cmnd - q[0]) + K_d * (cmnd_dot - q[1])

# Dynamics function
def qdot(t, q):
    return (A @ q + B.flatten() * u(q, t))

# Initial conditions & simulation parameters
t_span = (0, 10)   # time range
q0 = [0.0, 0.0]    # initial state: [x, xdot]

# Solve the ODE
solution = solve_ivp(qdot, t_span, q0, t_eval=np.linspace(*t_span, 400))

# Extract data
t = solution.t
x = solution.y[0]
xdot = solution.y[1]
F_vec = np.array([u([xi, vi], ti) for xi, vi, ti in zip(x, xdot, t)])

# --- Plot results ---
plt.figure()
plt.plot(t, x, label="x (m)")
plt.plot(t, xdot, label="dx/dt (m/s)")
plt.xlabel("time (s)")
plt.legend()
plt.grid(True)
plt.savefig("x_and_v_plot.png", dpi=150)

# --- Animation ---
fig, ax = plt.subplots()
ax.set_xlim(min(x)-1, max(x)+1)
ax.set_ylim(-1, 1)
cart, = ax.plot([], [], 'ks', markersize=20)  # cart as a square
force_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, color='red')
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def init():
    cart.set_data([], [])
    return cart, force_arrow, time_text

def update(frame):
    cart.set_data(x[frame], 0)
    ax.patches.clear()
    arrow_len = F_vec[frame] / 100.0
    ax.add_patch(plt.Arrow(x[frame], 0, arrow_len, 0, width=0.2, color='red'))
    time_text.set_text(f"t = {t[frame]:.2f} s")
    return cart, force_arrow, time_text

ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                    interval=25, blit=False)

ani.save("trolley_PD.gif", writer="pillow", fps=30)
plt.close()
