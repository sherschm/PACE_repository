from math import tau
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

class TwoDOFRobotDynamics:
    def __init__(self, m1=1.0, m2=0.5, l1=1.0, l2=0.8, I1=0.5, I2=0.2): # maybe add real equations on how to calculate the inertia
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.I1, self.I2 = I1, I2
        self.g = 9.81

    def dynamics(self, state,t):
        theta1, theta2, dtheta1, dtheta2 = state
        M11 = self.I1 + self.I2 + self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 +
              2*self.l1*(self.l2/2)*np.cos(theta2))
        M12 = self.I2 + self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*np.cos(theta2))
        M21 = M12
        M22 = self.I2 + self.m2*(self.l2/2)**2
        M = np.array([[M11, M12], [M21, M22]])


        h = self.m2*self.l1*(self.l2/2)*np.sin(theta2)
        C = np.array([
            [-h*dtheta2, -h*(dtheta2+dtheta1)],
            [h*dtheta1, 0]
        ])


        G = np.array([
            (self.m1*self.l1/2 + self.m2*self.l1)*self.g*np.cos(theta1) +
            self.m2*self.l2/2*self.g*np.cos(theta1 + theta2),
            self.m2*self.l2/2*self.g*np.cos(theta1 + theta2)
        ])


        tau = np.array([self.tau_1, self.tau_2]) # intial torque vector

        ddq = np.linalg.solve(M, tau - C @ np.array([dtheta1, dtheta2]) - G) #this is to solve the equation of motion
        #maybe try different ones
        return [dtheta1, dtheta2, ddq[0], ddq[1]]

    def simulate(self, t_span, initial_state, tau):
        self.tau_1,self.tau_2 = tau
        t = np.linspace(0, t_span, int(t_span*50))
        solution = odeint(self.dynamics, initial_state, t)
        return t, solution

    def animate_result(self, t, solution):
        plt.close('all')  # Close any existing figures
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-(self.l1 + self.l2 + 0.5), self.l1 + self.l2 + 0.5)
        ax.set_ylim(-(self.l1 + self.l2 + 0.5), self.l1 + self.l2 + 0.5)
        ax.grid(True)

        line, = ax.plot([], [], 'bo-', lw=2)
        time_template = 'Time = %.1fs'
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            theta1, theta2 = solution[i, 0], solution[i, 1]
            x1 = self.l1*np.cos(theta1)
            y1 = self.l1*np.sin(theta1)
            x2 = x1 + self.l2*np.cos(theta1 + theta2)
            y2 = y1 + self.l2*np.sin(theta1 + theta2)

            line.set_data([0, x1, x2], [0, y1, y2])
            time_text.set_text(time_template % t[i])
            return line, time_text

        anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                     interval=50, blit=True, init_func=init)
        anim.save('robot_animation.gif', writer='pillow', fps=20)
        print("Animation saved to robot_animation.gif")


robot = TwoDOFRobotDynamics()
initial_theta1 = 0
initial_theta2 = 0 
initial_state = [initial_theta1, initial_theta2, 0, 0]
tau_1 = 0 
tau_2 = 0 
tau = [tau_1, tau_2]
t, solution = robot.simulate(5, initial_state,tau)

robot.animate_result(t, solution)