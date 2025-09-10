import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.animation import FuncAnimation

# --- Trolley parameters ---
w_trol = 1.0
h_trol = 0.6
r_wheel = 0.125
sep_wheel = w_trol - 2 * r_wheel

def rectangle_patch(w, h, x, y, **kwargs):
    """Return a Rectangle patch."""
    return Rectangle((x, y), w, h, **kwargs)

def circle_patch(hc, kc, r, **kwargs):
    """Return a Circle patch."""
    return Circle((hc, kc), r, **kwargs)

def plot_trolley(ax, x):
    """Draw a trolley at horizontal position x."""
    patches = []
    # trolley body
    patches.append(rectangle_patch(w_trol, h_trol, x, 2*r_wheel,
                                   fill=False, edgecolor='black', linewidth=1.5))
    # left wheel
    patches.append(circle_patch(x + r_wheel, r_wheel, r_wheel,
                                edgecolor='blue', facecolor='lightblue', alpha=0.4))
    # right wheel
    patches.append(circle_patch(x + r_wheel + sep_wheel, r_wheel, r_wheel,
                                edgecolor='blue', facecolor='lightblue', alpha=0.4))
    for p in patches:
        ax.add_patch(p)
    return patches

def lin_interp(dataset, tvec, tvec_new):
    """Linear interpolation for animation time alignment."""
    dataset = np.asarray(dataset)
    out = np.empty((len(tvec_new), dataset.shape[1]))
    for i in range(dataset.shape[1]):
        out[:, i] = np.interp(tvec_new, tvec, dataset[:, i])
    return out

def animate_cart(x_sol, F_sol, t_vec, file_name):
    """
    Make an animation of trolleys with a force arrow on the first one.
    x_sol: (n_time, n_trolleys)
    F_sol: (n_time,)
    t_vec: (n_time,)
    file_name: output GIF filename
    """
    tstep_anim = 0.05
    t_anim = np.arange(tstep_anim, np.max(t_vec) + tstep_anim, tstep_anim)
    x_sol = np.atleast_2d(x_sol).T if np.ndim(x_sol) == 1 else np.array(x_sol)
    n_anim = len(t_anim)
    no_trolleys = x_sol.shape[1]
    
    x_anim = np.empty((n_anim, no_trolleys))
    x_anim[:, 0] = lin_interp(x_sol[:, [0]], t_vec, t_anim).flatten()
    for j in range(1, no_trolleys):
        x_anim[:, j] = lin_interp(x_sol[:, [j]], t_vec, t_anim).flatten() + 2 * w_trol * (j - 1)

    F_anim = np.interp(t_anim, t_vec, F_sol)

    fig, ax = plt.subplots()
    ax.set_xlim(np.min(x_sol) - 1, np.max(x_sol) + 1 + (no_trolleys) * (w_trol * 2))
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.axis('off')

    trolley_patches = []
    cable_lines = []
    force_arrow = None

    def init():
        ax.clear()
        ax.set_xlim(np.min(x_sol) - 1, np.max(x_sol) + 1 + (no_trolleys) * (w_trol * 2))
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        return []

    def update(i):
        ax.clear()
        
        # Set axis limits
        ax.set_xlim(np.min(x_sol) - 1, np.max(x_sol) + 1 + (no_trolleys) * (w_trol * 2))
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')

        # Show only bottom spine as the floor (x-axis)
        ax.spines['bottom'].set_position('zero')  # moves x-axis to y=0
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Show x ticks and labels
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        ax.tick_params(axis='x', direction='out', length=6)
        
        # Hide y ticks (optional)
        ax.yaxis.set_visible(False)

        # Draw trolleys and cables as before
        for j in range(no_trolleys):
            plot_trolley(ax, x_anim[i, j])
            if j > 0:
                ax.plot([x_anim[i, j-1] + w_trol, x_anim[i, j]],
                        [0.5*h_trol + 2*r_wheel, 0.5*h_trol + 2*r_wheel],
                        color='black', linewidth=1)

        # Draw force arrow
        if F_anim[i] >=0.001 or F_anim[i] <= -0.001:
            arrow_len = F_anim[i] / 10.0
            ax.add_patch(FancyArrow(x_anim[i, 0], 0.5*h_trol + 2*r_wheel,
                                    arrow_len, 0, width=0.05,
                                    head_width=0.15, head_length=0.15,
                                    color='red'))

        # Label the x-axis
        ax.set_xlabel("Position (m)")
        
        return ax.patches


    ani = FuncAnimation(fig, update, frames=n_anim, init_func=init,
                        blit=False, repeat=False, interval=1000*tstep_anim)
    ani.save(file_name, writer='pillow', fps=int(1/tstep_anim))
    plt.show()
    #plt.close(fig)

def plot_response(y, ydot, F, t, file_directory, filename="response_plot.png"):
    """
    Plot the cart response (position & velocity) and control input (force).
    
    Parameters
    ----------
    t : array-like
        Time vector.
    y : array-like
        Cart position (m).
    ydot : array-like
        Cart velocity (m/s).
    F : array-like
        Control input force (N).
    file_directory : str
        Directory where the plot will be saved.
    filename : str, optional
        Name of the output file (default: "response_plot.png").
    """
    
    fig, axs = plt.subplots(2, figsize=(8, 6), sharex=True, constrained_layout=True)

    # First subplot: Cart response
    axs[0].plot(t, y, label="y (m)")
    axs[0].plot(t, ydot, label="dy/dt (m/s)")
    axs[0].set_ylabel("Response")
    axs[0].set_title("Cart Response")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    # Second subplot: Control input
    axs[1].plot(t, F, label="Force u (N)", color="red")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Control Input")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    # Save & show
    save_path = str(file_directory) + "/" + filename
    plt.savefig(save_path, dpi=150)
    #plt.show()
    print(f"Plot saved to {save_path}")
