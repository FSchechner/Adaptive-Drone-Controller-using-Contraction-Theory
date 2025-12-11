import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from quadcopter_env import SimpleQuadcopter
from pd_controller import PDController
from ac_controller import AdaptiveController
from Drone import Drone
from scipy.optimize import minimize

class spiral_opt:
    def __init__(self, controller, drone_class=Drone, disturbance=None):
        self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.t = 0.0
        self.dt = 0.01
        self.r = 5
        self.env = SimpleQuadcopter(drone_class=drone_class, constant_disturbance=disturbance)
        self.controller = controller
        self.F = np.array([0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []

    def step(self):
        self.state = np.array([self.r*np.cos(self.t),
                               self.r*np.sin(self.t),
                               self.state[2]+self.state[5]*self.dt,
                               -self.r*np.sin(self.t),
                               self.r*np.cos(self.t),
                               2
                               ])
        self.t += self.dt

    def get_control(self):
        self.step()
        state = self.env.step(self.F, self.dt)
        pos = state[:3]
        vel = state[3:]
        self.F = self.controller.compute_control(pos, vel, self.state[:3], dt=self.dt)
        self.pos_hist.append(pos.copy())
        self.pos_d_hist.append(self.state[:3])

    def simulation(self):
        self.max_time = 20.0
        N = int(self.max_time / self.dt)
        for i in range(N):
            self.get_control()

    def get_error(self):
        pos = np.array(self.pos_hist)
        pos_d = np.array(self.pos_d_hist)
        error = pos - pos_d
        return np.mean(np.linalg.norm(error, axis=1))

    def reset(self):
        self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.t = 0.0
        self.env = SimpleQuadcopter(drone_class=Drone)
        self.F = np.array([0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []
        if hasattr(self.controller, 'reset'):
            self.controller.reset()

def optimize_pd():
    print("Optimizing PD Controller...")

    def objective(x):
        Kp_xy, Kd_xy, Kp_z, Kd_z, Ki_xy, Ki_z = x
        controller = PDController(Kp_xy=Kp_xy, Kd_xy=Kd_xy, Kp_z=Kp_z, Kd_z=Kd_z, Ki_xy=Ki_xy, Ki_z=Ki_z)
        sim = spiral_opt(controller)
        sim.simulation()
        error = sim.get_error()
        print(f"  Kp_xy={Kp_xy:.2f} Kd_xy={Kd_xy:.2f} Kp_z={Kp_z:.2f} Kd_z={Kd_z:.2f} Ki_xy={Ki_xy:.2f} Ki_z={Ki_z:.2f} -> error={error:.4f}")
        return error

    x0 = [29.89, 5.75, 2.56, 6.96, 0.0, 0.0]
    bounds = [(1, 30), (1, 15), (1, 30), (1, 15), (0, 5), (0, 5)]

    result = minimize(
        objective,
        x0,
        method='Powell',  # supports bounds
        bounds=bounds,
        options={'maxiter': 60, 'disp': True}
    )

    print(f"\nOptimal PD Gains:")
    print(f"  Kp_xy={result.x[0]:.4f}")
    print(f"  Kd_xy={result.x[1]:.4f}")
    print(f"  Kp_z={result.x[2]:.4f}")
    print(f"  Kd_z={result.x[3]:.4f}")
    print(f"  Ki_xy={result.x[4]:.4f}")
    print(f"  Ki_z={result.x[5]:.4f}")
    print(f"  Final error={result.fun:.4f}")

    return result.x

def optimize_ac():
    print("\nOptimizing Adaptive Controller...")

    class HeavyDrone:
        """Drone model with m=3.4 kg for robust AC tuning."""
        def __init__(self):
            self.Ixx = 0.028
            self.Iyy = 0.028
            self.Izz = 0.045
            self.m = 3.9
            self.F_max = 60
            self.tau_max = 4

    def objective(x):
        lambda_xy, lambda_z, k_xy, k_z, gamma_alpha, gamma_d = x
        controller = AdaptiveController(lambda_xy=lambda_xy, lambda_z=lambda_z,
                                       k_xy=k_xy, k_z=k_z,
                                       gamma_alpha=gamma_alpha, gamma_d=gamma_d)
        sim = spiral_opt(controller, drone_class=HeavyDrone, disturbance=[-5,3.3,0.1])
        sim.simulation()
        error = sim.get_error()
        print(f"  λxy={lambda_xy:.2f} λz={lambda_z:.2f} kxy={k_xy:.2f} kz={k_z:.2f} " +
              f"γα={gamma_alpha:.2f} γd={gamma_d:.2f} -> error={error:.4f}")
        return error

    x0 = [3.0, 4.0, 5.0, 8.0, 1.5, 0.5]
    bounds = [(0.5, 10), (0.5, 10), (1, 15), (1, 15), (0.1, 5), (0.1, 2)]

    result = minimize(
        objective,
        x0,
        method='Powell',  # supports bounds
        bounds=bounds,
        options={'maxiter': 20, 'disp': True}
    )

    print(f"\nOptimal AC Gains:")
    print(f"  lambda_xy={result.x[0]:.4f}")
    print(f"  lambda_z={result.x[1]:.4f}")
    print(f"  k_xy={result.x[2]:.4f}")
    print(f"  k_z={result.x[3]:.4f}")
    print(f"  gamma_alpha={result.x[4]:.4f}")
    print(f"  gamma_d={result.x[5]:.4f}")
    print(f"  Final error={result.fun:.4f}")

    return result.x

if __name__ == "__main__":
    pd_gains = optimize_pd()
    ac_gains = optimize_ac()
