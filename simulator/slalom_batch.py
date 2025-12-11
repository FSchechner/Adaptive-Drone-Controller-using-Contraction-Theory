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


def make_drone_class(mass):
    class CustomDrone:
        def __init__(self):
            self.Ixx = 0.028
            self.Iyy = 0.028
            self.Izz = 0.045
            self.m = mass
            self.F_max = 60
            self.tau_max = 4
    return CustomDrone


class Slalom:
    def __init__(self, controller, drone_class=Drone, disturbance=None):
        self.t = 0.0
        self.dt = 0.01
        self.T = 10.0
        self.Ax = 5.0
        self.kx = 1.2
        self.vy = 1.5
        self.z0 = 1.0
        self.Az = 1.0
        self.kz = 0.8

        self.env = SimpleQuadcopter(drone_class=drone_class, constant_disturbance=disturbance)
        self.controller = controller
        self.F = np.array([0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []

    def step_reference(self):
        x = self.Ax * np.sin(self.kx * self.t)
        y = self.vy * self.t
        z = self.z0 + self.Az * np.sin(self.kz * self.t)

        dx = self.Ax * self.kx * np.cos(self.kx * self.t)
        dy = self.vy
        dz = self.Az * self.kz * np.cos(self.kz * self.t)

        ddx = -self.Ax * (self.kx ** 2) * np.sin(self.kx * self.t)
        ddy = 0.0
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
        mean_total = np.mean(np.linalg.norm(error, axis=1))
        mean_x = np.mean(np.abs(error[:, 0]))
        mean_y = np.mean(np.abs(error[:, 1]))
        mean_z = np.mean(np.abs(error[:, 2]))
        return mean_total, mean_x, mean_y, mean_z, pos, pos_d, error


def run_scenario(controller, drone_class, disturbance):
    sim = Slalom(controller, drone_class=drone_class, disturbance=disturbance)
    sim.run()
    return sim.errors()


def main():
    scenarios = []
    scenarios.append(("Baseline (m=1.9, no wind)", Drone, [0.0, 0.0, 0.0]))
    mass_values = np.linspace(1.9, 4.0, 5)
    for m in mass_values:
        scenarios.append((f"Mass {m:.2f} kg, no wind", make_drone_class(m), [0.0, 0.0, 0.0]))
    wind_values = np.linspace(0.0, 10.0, 5)
    wind_vec = np.array([-2.0, 4.0, 0.0])
    for w in wind_values:
        wvec = w * wind_vec / np.linalg.norm(wind_vec) if w != 0 else wind_vec * 0.0
        scenarios.append((f"Wind {wvec}", Drone, wvec.tolist()))
    for m, w in zip(mass_values, wind_values):
        wvec = w * wind_vec / np.linalg.norm(wind_vec) if w != 0 else wind_vec * 0.0
        scenarios.append((f"Mass {m:.2f} kg + Wind {wvec}", make_drone_class(m), wvec.tolist()))

    ac_errors = {'base': [], 'mass': [], 'wind': [], 'comb': []}
    pd_errors = {'base': [], 'mass': [], 'wind': [], 'comb': []}

    labels = []
    for idx, (label, drone_cls, wind) in enumerate(scenarios, start=1):
        print(f"Run {idx:02d}: {label}")
        ac_res = run_scenario(AdaptiveController(), drone_cls, wind)
        pd_res = run_scenario(PDController(), drone_cls, wind)
        labels.append(label)
        ac_total = ac_res[0]; pd_total = pd_res[0]
        print(f"  AC total error: {ac_total:.3f}")
        print(f"  PD total error: {pd_total:.3f}\n")

        if idx == 1:
            ac_errors['base'].append(ac_total)
            pd_errors['base'].append(pd_total)
        elif 2 <= idx <= 6:
            ac_errors['mass'].append(ac_total)
            pd_errors['mass'].append(pd_total)
        elif 7 <= idx <= 11:
            ac_errors['wind'].append(ac_total)
            pd_errors['wind'].append(pd_total)
        else:
            ac_errors['comb'].append(ac_total)
            pd_errors['comb'].append(pd_total)

    # Prepare x-axes
    x_base = np.arange(1, 6)
    x_mass = np.arange(1, len(ac_errors['mass']) + 1)
    x_wind = np.arange(1, len(ac_errors['wind']) + 1)
    x_comb = np.arange(1, len(ac_errors['comb']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_base, np.full_like(x_base, ac_errors['base'][0], dtype=float), 'b--', label='AC baseline')
    plt.plot(x_base, np.full_like(x_base, pd_errors['base'][0], dtype=float), 'r--', label='PD baseline')
    plt.plot(x_mass, ac_errors['mass'], 'bo-', label='AC mass')
    plt.plot(x_mass, pd_errors['mass'], 'ro-', label='PD mass')
    plt.plot(x_wind, ac_errors['wind'], 'bs-', label='AC wind')
    plt.plot(x_wind, pd_errors['wind'], 'rs-', label='PD wind')
    plt.plot(x_comb, ac_errors['comb'], 'b^-', label='AC combined')
    plt.plot(x_comb, pd_errors['comb'], 'r^-', label='PD combined')
    plt.xlabel('Run index within category')
    plt.ylabel('Tracking error (mean total, m)')
    plt.title('Slalom Tracking Error Across Scenarios')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('slalom_batch_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
