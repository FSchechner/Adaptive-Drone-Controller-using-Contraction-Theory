import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from quadcopter_env import SimpleQuadcopter
from pd_controller import PDController
from Drone import Drone
from scipy.optimize import minimize

class spiral_opt_pd:
    def __init__(self, controller, drone_class=Drone):
        self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.t = 0.0
        self.dt = 0.01
        self.r = 5
        self.env = SimpleQuadcopter(drone_class=drone_class)
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
                               2])
        self.t += self.dt

    def get_control(self):
        self.step()
        state = self.env.step(self.F, self.dt)
        pos = state[:3]
        vel = state[3:]
        self.F = self.controller.compute_control(pos, vel, self.state[:3], dt=self.dt)
        self.pos_hist.append(pos.copy())
        self.pos_d_hist.append(self.state[:3])

    def simulation(self, max_time=20.0):
        N = int(max_time / self.dt)
        for _ in range(N):
            self.get_control()

    def get_error(self):
        pos = np.array(self.pos_hist)
        pos_d = np.array(self.pos_d_hist)
        error = pos - pos_d
        return np.mean(np.linalg.norm(error, axis=1))


def optimize_pd():
    print("Optimizing PD Controller (trajectory tracking)...")

    def objective(x):
        Kp_xy, Kd_xy, Kp_z, Kd_z, Ki_xy, Ki_z = x
        controller = PDController(Kp_xy=Kp_xy, Kd_xy=Kd_xy, Kp_z=Kp_z, Kd_z=Kd_z, Ki_xy=Ki_xy, Ki_z=Ki_z)
        sim = spiral_opt_pd(controller)
        sim.simulation()
        error = sim.get_error()
        print(f"  Kp_xy={Kp_xy:.2f} Kd_xy={Kd_xy:.2f} Kp_z={Kp_z:.2f} Kd_z={Kd_z:.2f} Ki_xy={Ki_xy:.2f} Ki_z={Ki_z:.2f} -> error={error:.4f}")
        return error

    x0 = [30.0, 8.6, 30.0, 8.5, 0.0, 0.07]
    bounds = [(5, 50), (1, 20), (5, 50), (1, 20), (0, 1), (0, 1)]

    result = minimize(
        objective,
        x0,
        method='Powell',
        bounds=bounds,
        options={'maxiter': 40, 'disp': True}
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

if __name__ == "__main__":
    optimize_pd()
