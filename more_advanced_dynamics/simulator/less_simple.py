import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from cascaded_quadcopter_env import CascadedQuadcopter
from pd_controller import PDController
from ac_controller import AdaptiveController
from attitude_controller import AttitudeController
from Drone import Drone, Drone_with_Package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class spiral:
    def __init__(self, controller, drone_class=None, disturbance=None):
        self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.t = 0.0
        self.dt = 0.01
        self.r = 5
        self.env = CascadedQuadcopter(drone_class=drone_class, constant_disturbance=disturbance)
        self.pos_controller = controller
        self.att_controller = AttitudeController()
        self.u = np.array([0.0, 0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []

    def step(self):
        if self.t < 20:
            self.state = np.array([self.r*np.cos(self.t),
                               self.r*np.sin(self.t),
                               self.state[2]+self.state[5]*self.dt,
                               -self.r*np.sin(self.t),
                               self.r*np.cos(self.t),
                               2
                               ])
        else: 
            if self.t < 25: 
                self.state = np.array([self.r/2*np.cos(self.t*2),
                               self.state[1]+self.state[4]*self.dt,
                               self.state[2]+self.state[5]*self.dt,
                               2,
                               2,
                               2
                               ])

            else: 
                self.state = np.array([self.state[0]+self.state[3]*self.dt,
                               self.state[1]+self.state[4]*self.dt,
                               self.state[2]+self.state[5]*self.dt,
                               0,
                               0,
                               0
                               ])

        self.t += self.dt

        

    def get_control(self):
        # Update reference trajectory
        self.step()

        # Position control runs at dt, attitude at dt/3
        # Pattern: Pos → Att → Dynamics → Att → Dynamics → Att → Dynamics → Pos → ...
        dt_att = self.dt / 3.0

        # Get state and compute position control
        full_state = self.env.get_state()
        pos = full_state[0:3]
        vel = full_state[3:6]
        F_des = self.pos_controller.compute_control(pos, vel, self.state[:3], dt=self.dt)

        # Run 3 attitude control cycles at higher rate
        for i in range(3):
            full_state = self.env.get_state()
            self.u = self.att_controller.compute_control(F_des, full_state)
            full_state = self.env.step(self.u, dt_att)

        # Record final position
        pos = full_state[0:3]
        self.pos_hist.append(pos.copy())
        self.pos_d_hist.append(self.state[:3])

    def simulation(self):
        self.max_time = 30.0
        self.dt = 0.01
        N = int(self.max_time / self.dt)
        for i in range(N):
            self.get_control()
        return None, None, None

    def get_errors(self):
        pos = np.array(self.pos_hist)
        pos_d = np.array(self.pos_d_hist)
        error = pos - pos_d

        mean_error_x = np.mean(np.abs(error[:, 0]))
        mean_error_y = np.mean(np.abs(error[:, 1]))
        mean_error_z = np.mean(np.abs(error[:, 2]))
        mean_error_total = np.mean(np.linalg.norm(error, axis=1))

        return mean_error_x, mean_error_y, mean_error_z, mean_error_total, pos, pos_d, error

def plot_comparison(results_list, labels):
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for (_, _, _, _, pos, pos_d, _), label in zip(results_list, labels):
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label, linewidth=1.5)
    ax1.plot(pos_d[:, 0], pos_d[:, 1], pos_d[:, 2], 'k--', label='Desired', linewidth=2)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    for idx, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(2, 3, 2 + idx)
        for (_, _, _, _, _, _, error), label in zip(results_list, labels):
            ax.plot(error[:, idx], label=label, linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel(f'{axis_name} Error [m]')
        ax.set_title(f'{axis_name} Error')
        ax.legend()
        ax.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    for (_, _, _, _, _, _, error), label in zip(results_list, labels):
        ax5.plot(np.linalg.norm(error, axis=1), label=label, linewidth=1)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Total Error [m]')
    ax5.set_title('Total Error')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    for (_, _, _, _, pos, pos_d, _), label in zip(results_list, labels):
        ax6.plot(pos[:, 0], pos[:, 1], label=label, linewidth=1.5)
    ax6.plot(pos_d[:, 0], pos_d[:, 1], 'k--', label='Desired', linewidth=2)
    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('y [m]')
    ax6.set_title('XY Trajectory')
    ax6.legend()
    ax6.grid(True)
    ax6.axis('equal')

    plt.tight_layout()
    plt.savefig('cascaded_result.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    results = []
    labels = []

    disturbance = [1.0, 0.0, 0.0]

    print("AC + Drone")
    sim1 = spiral(AdaptiveController(), Drone, disturbance)
    sim1.simulation()
    results.append(sim1.get_errors())
    labels.append('AC + Drone')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    print("PD + Drone")
    sim2 = spiral(PDController(), Drone, disturbance)
    sim2.simulation()
    results.append(sim2.get_errors())
    labels.append('PD + Drone')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    print("AC + Drone_with_Package")
    sim3 = spiral(AdaptiveController(), Drone_with_Package, disturbance)
    sim3.simulation()
    results.append(sim3.get_errors())
    labels.append('AC + Package')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    print("PD + Drone_with_Package")
    sim4 = spiral(PDController(), Drone_with_Package, disturbance)
    sim4.simulation()
    results.append(sim4.get_errors())
    labels.append('PD + Package')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    plot_comparison(results, labels)
    print("Plot saved to cascaded_result.png")
