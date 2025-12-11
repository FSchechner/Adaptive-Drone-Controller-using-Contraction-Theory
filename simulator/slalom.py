import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from quadcopter_env import SimpleQuadcopter
from pd_controller import PDController
from ac_controller import AdaptiveController
from Drone import Drone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Slalom:
    def __init__(self, controller, drone_class=Drone, disturbance=None):
        self.t = 0.0
        self.dt = 0.01
        self.T = 20.0
        self.Ax = 5.0  # x weave amplitude
        self.kx = 1.2  # x weave frequency
        self.vy0 = 1.5  # nominal forward speed in y
        self.vy_amp = 0.8  # speed variation amplitude
        self.vy_freq = 0.5  # speed variation frequency
        self.z0 = 1.0  # base altitude
        self.Az = 1.0  # z weave amplitude
        self.kz = 0.8  # z weave frequency

        self.env = SimpleQuadcopter(drone_class=drone_class, constant_disturbance=disturbance)
        self.controller = controller
        self.F = np.array([0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []

    def step_reference(self):
        x = self.Ax * np.sin(self.kx * self.t)
        vy = self.vy0 + self.vy_amp * np.sin(self.vy_freq * self.t)
        ay = self.vy_amp * self.vy_freq * np.cos(self.vy_freq * self.t)  # derivative of vy

        y = vy * self.t  # approximate forward integration
        z = self.z0 + self.Az * np.sin(self.kz * self.t)

        dx = self.Ax * self.kx * np.cos(self.kx * self.t)
        dy = vy
        dz = self.Az * self.kz * np.cos(self.kz * self.t)

        ddx = -self.Ax * (self.kx ** 2) * np.sin(self.kx * self.t)
        ddy = ay
        ddz = -self.Az * (self.kz ** 2) * np.sin(self.kz * self.t)

        self.t += self.dt
        pos_d = np.array([x, y, z])
        vel_d = np.array([dx, dy, dz])
        acc_d = np.array([ddx, ddy, ddz])
        return pos_d, vel_d, acc_d

    def get_control(self):
        pos_d, vel_d, acc_d = self.step_reference()
        state = self.env.step(self.F, self.dt)
        pos = state[:3]
        vel = state[3:]
        self.F = self.controller.compute_control(pos, vel, pos_d, vel_d, acc_d, dt=self.dt)
        self.pos_hist.append(pos.copy())
        self.pos_d_hist.append(pos_d.copy())

    def run(self):
        N = int(self.T / self.dt)
        for _ in range(N):
            self.get_control()

    def errors(self):
        pos = np.array(self.pos_hist)
        pos_d = np.array(self.pos_d_hist)
        error = pos - pos_d
        mean_error = np.mean(np.linalg.norm(error, axis=1))
        mean_error_x = np.mean(np.abs(error[:, 0]))
        mean_error_y = np.mean(np.abs(error[:, 1]))
        mean_error_z = np.mean(np.abs(error[:, 2]))
        return mean_error, mean_error_x, mean_error_y, mean_error_z, pos, pos_d, error


def plot_results(results, labels, filename):
    plt.figure(figsize=(10, 6))
    for (mean_error, _, _, _, pos, pos_d, _), label in zip(results, labels):
        plt.plot(np.linalg.norm(pos - pos_d, axis=1), label=f"{label} total error")
    plt.xlabel('Step')
    plt.ylabel('Tracking error [m]')
    plt.title('Slalom Tracking Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    # 3D trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for (_, _, _, _, pos, pos_d, _), label in zip(results, labels):
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label)
    ax.plot(pos_d[:, 0], pos_d[:, 1], pos_d[:, 2], 'k--', label='Desired')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Slalom Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.savefig('slalom_trajectory.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    runs = [
        ("AC", AdaptiveController()),
        ("PD", PDController()),
    ]
    results = []
    labels = []
    for label, controller in runs:
        sim = Slalom(controller)
        sim.run()
        res = sim.errors()
        results.append(res)
        labels.append(label)
        mean_total, mx, my, mz, *_ = res
        print(f"{label}: total={mean_total:.3f} x={mx:.3f} y={my:.3f} z={mz:.3f}")
    plot_results(results, labels, 'slalom_error.png')
    print("Plots saved: slalom_error.png, slalom_trajectory.png")
