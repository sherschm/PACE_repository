import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from KinematicsFuncs import calc_R20, Rz
import matplotlib.pyplot as plt
from PendulumParameters import l1, l2

def L1_tip_pos(theta1):
    return Rz(theta1).T @ np.array([0, l1, 0])

def L2_tip_pos(theta1, theta2):
    return calc_R20(theta1, theta2).T @ np.array([0, l1, -l2])

def plot_rot_pendulum(q, i, dt, ax):
    coords = np.column_stack([
        np.zeros(3),
        L1_tip_pos(q[0]),
        L2_tip_pos(q[0], q[1])
    ])
    ax.clear()
    ax.plot(coords[0, :], coords[1, :], coords[2, :], linewidth=8)
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)
    ax.set_title(f"Time = {np.floor(i*dt)} s")

def general_lin_interp(dataset, tvec, tvec_new):
    # dataset: N x X, N=time, X=states
    interped_data = np.zeros((len(tvec_new), dataset.shape[1]))
    for i in range(dataset.shape[1]):
        interp_fn = interp1d(tvec, dataset[:, i], kind='linear', fill_value='extrapolate')
        interped_data[:, i] = interp_fn(tvec_new)
    return interped_data

def rot_pendulum_animator(x_sol, name="rotary_pendulum_anim"):
    print("Creating animation...")

    tvec = x_sol.t
    anim_fps = 20
    tvec_anim = np.arange(1/anim_fps, tvec[-1], 1/anim_fps)
    np.arange(1/anim_fps, tvec[-1], 1/anim_fps)
    #x_anim = general_lin_interp(x_sol, tvec, tvec_anim)

    # resample state trajectory at animation fps using cubic interpolation
    x_anim = interp1d(tvec, x_sol.y, kind='cubic', axis=1)(tvec_anim)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        plot_rot_pendulum(x_anim[:,i], i, 1/anim_fps, ax)
        ax.set_title(f"Time = {i/anim_fps:.2f} s")

    anim = FuncAnimation(fig, update, frames=len(tvec_anim), interval=1000/anim_fps)
    anim.save(f"{name}.gif", writer='pillow', fps=anim_fps)
    plt.show()
    #plt.close(fig)

    
def plot_response(sol, F, file_directory, filename="response_plot.png"): 

    t= list(sol.t)

    fig, axs = plt.subplots(3, figsize=(8, 6), sharex=True, constrained_layout=True)

    # First subplot: Cart response
    axs[0].plot(list(sol.t), list(sol.y[0]), label='θ1')
    axs[0].plot(list(sol.t), list(sol.y[1]), label='θ2')
    axs[0].set_ylabel("Angle (rad)")
    axs[0].set_title("Pendulum Position Response")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].plot(list(sol.t), list(sol.y[2]), label='dθ1/dt')
    axs[1].plot(list(sol.t), list(sol.y[3]), label='dθ2/dt')
    axs[1].set_ylabel("Angular velocity (rad/s)")
    axs[1].set_title("Pendulum Velocity Response")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    # Second subplot: Control input
    axs[2].plot(t, F, color="red")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("θ1 Acceleration (rad/s^2)")
    axs[2].set_title("Control acceleration")
    axs[2].grid(True)

    # Save & show
    save_path = str(file_directory) + "/" + filename
    plt.savefig(save_path, dpi=150)
    #plt.show()
    print(f"Plot saved to {save_path}")
