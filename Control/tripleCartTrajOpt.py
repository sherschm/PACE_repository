import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# System parameters
m1, m2, m3 = 10, 10, 10
k1, k2, k3 = 30, 30, 30
M = np.diag([m1, m2, m3])
K = np.array([[ k1,   -k1,    0],
              [-k1, k1+k2, -k2],
              [  0,  -k2,   k2]])

tmax = 10
tstep = 0.05
tvec = np.arange(0, tmax, tstep)
n_sim = len(tvec)

cmnd = 7.0

# Dynamics function
def dynamics(t, state, F_func):
    x = state[:3]
    v = state[3:]
    F = np.array([F_func(t), 0, 0])  # only cart 1 actuated
    a = np.linalg.inv(M) @ (F - K @ x)
    return np.hstack([v, a])

# Objective: track cart 3 position to cmnd
def objective(F_guess):
    # Interpolate force trajectory
    def F_func(t):
        return np.interp(t, tvec, F_guess)
    # Simulate system
    sol = solve_ivp(dynamics, [0, tmax], np.zeros(6), t_eval=tvec, args=(F_func,))
    x3 = sol.y[2, :] + 4*0.5  # add trolley offset
    return np.sum((x3 - cmnd)**2)

# Initial guess: zero force
F0 = np.zeros(n_sim)

# Optimize with bounds
bounds = [(-100, 100)] * n_sim
res = minimize(objective, F0, bounds=bounds, method="SLSQP", options={"maxiter": 100, "disp": True})

# Extract solution
F_opt = res.x
def F_func(t): return np.interp(t, tvec, F_opt)
sol = solve_ivp(dynamics, [0, tmax], np.zeros(6), t_eval=tvec, args=(F_func,))

x_matrix = sol.y[:3, :]
F_matrix = F_opt

# Plot
plt.figure()
plt.plot(tvec, x_matrix[0,:], label="Cart 1")
plt.plot(tvec, x_matrix[1,:], label="Cart 2")
plt.plot(tvec, x_matrix[2,:], label="Cart 3")
plt.axhline(cmnd, color='k', linestyle='--', label="Desired")
plt.legend()
plt.show()

plt.figure()
plt.plot(tvec, F_matrix, label="Optimal Force")
plt.legend()
plt.show()
