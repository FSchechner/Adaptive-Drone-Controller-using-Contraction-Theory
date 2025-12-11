import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from quadcopter_env import SimpleQuadcopter
from pd_controller import PDController
from ac_controller import AdaptiveController
from Drone import Drone, Drone_with_Package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class spiral:
    def __init__(self, controller, drone_class=None, disturbance=None):
        self.t = 0.0
        self.dt = 0.01
        self.r = 5.0
        self.omega = 1.0
        self.vz_spiral = 2.0
        self.a_linear = np.array([2.0, 2.0, 2.0])

        self.env = SimpleQuadcopter(drone_class=drone_class, constant_disturbance=disturbance)
        self.controller = controller
        self.F = np.array([0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []

        # Pre-compute boundary conditions for piecewise trajectory
        self.pos20, self.vel20, _ = self._spiral_state(20.0)
        tau_linear = 5.0
        self.pos25 = self.pos20 + self.vel20 * tau_linear + 0.5 * self.a_linear * (tau_linear ** 2)

    def _spiral_state(self, t):
        """Desired pos/vel/acc for the spiral phase (0-20s)."""
        pos = np.array([
            self.r * np.cos(self.omega * t),
            self.r * np.sin(self.omega * t),
            self.vz_spiral * t
        ])
        vel = np.array([
            -self.r * self.omega * np.sin(self.omega * t),
            self.r * self.omega * np.cos(self.omega * t),
            self.vz_spiral
        ])
        acc = np.array([
            -self.r * (self.omega ** 2) * np.cos(self.omega * t),
            -self.r * (self.omega ** 2) * np.sin(self.omega * t),
            0.0
        ])
        return pos, vel, acc

    def step_reference(self):
        """Generate desired position, velocity, and acceleration."""
        if self.t < 20.0:
            pos_d, vel_d, acc_d = self._spiral_state(self.t)
        elif self.t < 25.0:
            tau = self.t - 20.0
            pos_d = self.pos20 + self.vel20 * tau + 0.5 * self.a_linear * (tau ** 2)
            vel_d = self.vel20 + self.a_linear * tau
            acc_d = self.a_linear
        else:
            pos_d = self.pos25
            vel_d = np.array([0.0, 0.0, 0.0])
            acc_d = np.array([0.0, 0.0, 0.0])

        self.t += self.dt
        return pos_d, vel_d, acc_d

    def get_control(self):
        pos_d, vel_d, acc_d = self.step_reference()
        state = self.env.step(self.F, self.dt)
        pos = state[:3]
        vel = state[3:]
        self.F = self.controller.compute_control(pos, vel, pos_d, vel_d, acc_d, dt=self.dt)
        self.pos_hist.append(pos.copy())
        self.pos_d_hist.append(pos_d.copy())

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
    plt.savefig('comparison_result.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    results = []
    labels = []

    # Test 1: Baseline (no disturbance, nominal mass)
    print("=== Baseline: Nominal Mass, No Disturbance ===")
    print("AC + Drone (Baseline)")
    sim1 = spiral(AdaptiveController(), Drone, None)
    sim1.simulation()
    results.append(sim1.get_errors())
    labels.append('AC Baseline')
    ex, ey, ez, et = results[-1][:4]
    est = sim1.controller.get_estimates()
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}")
    print(f"Estimates: m_hat={est['m_hat']:.3f} kg, d_hat=[{est['d_hat'][0]:.3f}, {est['d_hat'][1]:.3f}, {est['d_hat'][2]:.3f}] m/s²\n")

    print("PD + Drone (Baseline)")
    sim2 = spiral(PDController(), Drone, None)
    sim2.simulation()
    results.append(sim2.get_errors())
    labels.append('PD Baseline')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    # Test 2: Nominal mass + 3N disturbance
    disturbance = [7.0, 0.0, 0.0]
    print("=== Test 1: Nominal Mass + 3N Disturbance ===")
    print("AC + Drone + Disturbance")
    sim3 = spiral(AdaptiveController(), Drone, disturbance)
    sim3.simulation()
    results.append(sim3.get_errors())
    labels.append('AC + Wind')
    ex, ey, ez, et = results[-1][:4]
    est = sim3.controller.get_estimates()
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}")
    print(f"Estimates: m_hat={est['m_hat']:.3f} kg, d_hat=[{est['d_hat'][0]:.3f}, {est['d_hat'][1]:.3f}, {est['d_hat'][2]:.3f}] m/s²\n")

    print("PD + Drone + Disturbance")
    sim4 = spiral(PDController(), Drone, disturbance)
    sim4.simulation()
    results.append(sim4.get_errors())
    labels.append('PD + Wind')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    # Test 3: Heavy mass + 3N disturbance
    print("=== Test 2: Heavy Mass + 3N Disturbance ===")
    print("AC + Package + Disturbance")
    sim5 = spiral(AdaptiveController(), Drone_with_Package, disturbance)
    sim5.simulation()
    results.append(sim5.get_errors())
    labels.append('AC + Package + Wind')
    ex, ey, ez, et = results[-1][:4]
    est = sim5.controller.get_estimates()
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}")
    print(f"Estimates: m_hat={est['m_hat']:.3f} kg, d_hat=[{est['d_hat'][0]:.3f}, {est['d_hat'][1]:.3f}, {est['d_hat'][2]:.3f}] m/s²\n")

    print("PD + Package + Disturbance")
    sim6 = spiral(PDController(), Drone_with_Package, disturbance)
    sim6.simulation()
    results.append(sim6.get_errors())
    labels.append('PD + Package + Wind')
    ex, ey, ez, et = results[-1][:4]
    print(f"Errors: x={ex:.3f} y={ey:.3f} z={ez:.3f} total={et:.3f}\n")

    plot_comparison(results, labels)
    print("Plot saved to comparison_result.png")
