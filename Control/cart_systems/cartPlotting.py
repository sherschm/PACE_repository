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
    Animate multiple carts with a force arrow on the first one.
    
    Parameters
    ----------
    x_sol : (n_time, n_trolleys) array
        Time evolution of trolley positions.
    F_sol : (n_time,) array
        Force applied to the first trolley over time.
    t_vec : (n_time,) array
        Time vector corresponding to x_sol and F_sol.
    file_name : str
        Output GIF filename.
    """
    tstep_anim = 0.05
    t_anim = np.arange(tstep_anim, np.max(t_vec) + tstep_anim, tstep_anim)
    x_sol = np.atleast_2d(x_sol).T if np.ndim(x_sol) == 1 else np.array(x_sol)
    n_anim = len(t_anim)
    no_trolleys = x_sol.shape[0] if x_sol.shape[0] < x_sol.shape[1] else x_sol.shape[1]

    # Linear interpolation of positions
    x_anim = np.empty((n_anim, no_trolleys))
    for j in range(no_trolleys):
        x_anim[:, j] = np.interp(t_anim, t_vec, x_sol[j, :]) + 2 * w_trol * j

    # Interpolated force
    F_anim = np.interp(t_anim, t_vec, F_sol)

    fig, ax = plt.subplots(figsize=(8, 3))

    def init():
        ax.set_xlim(np.min(x_anim) - 1, np.max(x_anim) + 2*w_trol)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        return []

    def update(i):
        ax.clear()
        ax.set_xlim(np.min(x_anim) - 1, np.max(x_anim) + 2*w_trol)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')

        # Draw carts and springs
        for j in range(no_trolleys):
            # trolley rectangle
            ax.add_patch(plt.Rectangle((x_anim[i, j], r_wheel*2),
                                       w_trol, h_trol,
                                       fc='skyblue', ec='black'))
            # wheels
            ax.add_patch(plt.Circle((x_anim[i, j] + 0.2*w_trol, r_wheel), r_wheel, fc='k'))
            ax.add_patch(plt.Circle((x_anim[i, j] + 0.8*w_trol, r_wheel), r_wheel, fc='k'))
            
            # connecting spring (line) between this cart and the previous
            if j > 0:
                ax.plot([x_anim[i, j-1] + w_trol, x_anim[i, j]],
                        [0.5*h_trol + 2*r_wheel, 0.5*h_trol + 2*r_wheel],
                        color='black', linewidth=2)

        # Draw force arrow on cart 1
        if abs(F_anim[i]) > 1e-3:
            arrow_len = F_anim[i] / 20.0
            ax.add_patch(FancyArrow(x_anim[i, 0], h_trol/2 + 2*r_wheel,
                                    arrow_len, 0,
                                    width=0.05,
                                    head_width=0.15, head_length=0.15,
                                    color='red'))

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

def animate_cartpole(x_sol, F_sol, t_vec, file_name, L=1.0):
    """
    Animate a cart-pole system.
    
    x_sol: (n_time, 4) -> [y, theta, y_dot, theta_dot]
    F_sol: (n_time,)   -> applied force on cart
    t_vec: (n_time,)   -> time vector
    file_name: str     -> output GIF filename
    L: float           -> pendulum length (default 1.0)
    """
    tstep_anim = 0.05
    t_anim = np.arange(tstep_anim, np.max(t_vec) + tstep_anim, tstep_anim)

    # interpolate solutions to animation timeline
    y_anim     = lin_interp(x_sol[:, [0]], t_vec, t_anim).flatten()
    theta_anim = lin_interp(x_sol[:, [1]], t_vec, t_anim).flatten()
    F_anim     = np.interp(t_anim, t_vec, F_sol)

    n_anim = len(t_anim)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    def init():
        ax.clear()
        ax.set_xlim(np.min(y_anim) - 2, np.max(y_anim) + 2)
        ax.set_ylim(-L - 0.5, L + 1.5)
        ax.set_aspect('equal')
        return []

    def update(i):
        ax.clear()

        # Axis limits
        ax.set_xlim(np.min(y_anim) - 2, np.max(y_anim) + 2)
        ax.set_ylim(-L - 0.5, L + 1.5)
        ax.set_aspect('equal')

        # Draw floor
        ax.axhline(0, color='k', linewidth=1)

        # Cart geometry
        cart_x = y_anim[i]
        cart_y = 0.25   # height of cart bottom above ground
        plot_trolley(ax, cart_x)  # uses your existing function

        # Pendulum coordinates
        theta = theta_anim[i]
        cart_center = (cart_x + w_trol/2, cart_y + h_trol/2)  # center of cart
        px = cart_center[0] + L * np.sin(theta)
        py = cart_center[1] - L * np.cos(theta)

        # Draw pendulum rod and bob
        ax.plot([cart_center[0], px], [cart_center[1], py], 'k-', lw=2)
        ax.add_patch(plt.Circle((px, py), 0.1, fc='blue'))

        # Force arrow
        if abs(F_anim[i]) > 1e-3:
            arrow_len = F_anim[i] / 10.0
            ax.add_patch(FancyArrow(cart_x, cart_y + h_trol/2,
                                    arrow_len, 0, width=0.05,
                                    head_width=0.15, head_length=0.15,
                                    color='red'))

        # Label
        ax.set_xlabel("Position (m)")

        return ax.patches

    ani = FuncAnimation(fig, update, frames=n_anim, init_func=init,
                        blit=False, repeat=False, interval=1000*tstep_anim)
    ani.save(file_name, writer='pillow', fps=int(1/tstep_anim))
    #plt.show()


def plot_response_cartpole(solution, F, t, file_directory, filename="response_plot.png"):
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
    axs[0].plot(t, solution.y[0], label="y (m)")
    axs[0].plot(t, solution.y[1], label="theta (rad)")
    axs[0].plot(t, solution.y[2], label="ydot (m/2)")
    axs[0].plot(t, solution.y[3], label="theta_dot (rad/s)")
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