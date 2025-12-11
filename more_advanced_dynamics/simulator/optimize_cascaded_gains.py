import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from cascaded_quadcopter_env import CascadedQuadcopter
from pd_controller import PDController
from ac_controller import AdaptiveController
from attitude_controller import AttitudeController
from Drone import Drone
from scipy.optimize import minimize

class spiral_opt:
    def __init__(self, pos_controller, drone_class=Drone):
        self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.t = 0.0
        self.dt = 0.01
        self.r = 5
        self.env = CascadedQuadcopter(drone_class=drone_class)
        self.pos_controller = pos_controller
        self.att_controller = AttitudeController()
        self.u = np.array([0.0, 0.0, 0.0, 0.0])
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
        # Update reference trajectory
        self.step()

        # Get current state and compute position control (runs at dt)
        full_state = self.env.get_state()
        pos = full_state[0:3]
        vel = full_state[3:6]
        F_des = self.pos_controller.compute_control(pos, vel, self.state[:3], dt=self.dt)

        # Run attitude controller 3 times faster (dt/3)
        dt_att = self.dt / 3.0
        for i in range(3):
            full_state = self.env.get_state()
            self.u = self.att_controller.compute_control(F_des, full_state)
            full_state = self.env.step(self.u, dt_att)

        # Record final position after 3 attitude steps
        pos = full_state[0:3]
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
        self.env = CascadedQuadcopter(drone_class=Drone)
        self.u = np.array([0.0, 0.0, 0.0, 0.0])
        self.pos_hist = []
        self.pos_d_hist = []
        if hasattr(self.pos_controller, 'reset'):
            self.pos_controller.reset()

def optimize_pd():
    print("Optimizing PD Controller for Cascaded System...")

    def objective(x):
        Kp_xy, Kd_xy, Kp_z, Kd_z = x
        controller = PDController(Kp_xy=Kp_xy, Kd_xy=Kd_xy, Kp_z=Kp_z, Kd_z=Kd_z, Ki_xy=0.0, Ki_z=0.0)
        sim = spiral_opt(controller)
        try:
            sim.simulation()
            error = sim.get_error()
        except:
            error = 1000.0
        print(f"  Kp_xy={Kp_xy:.2f} Kd_xy={Kd_xy:.2f} Kp_z={Kp_z:.2f} Kd_z={Kd_z:.2f} -> error={error:.4f}")
        return error

    x0 = [29.83, 8.78, 11.12, 13.81]
    bounds = [(5, 50), (1, 20), (5, 50), (1, 20)]

    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                     options={'maxiter': 100, 'disp': True})

    print(f"\nOptimal PD Gains for Cascaded System:")
    print(f"  Kp_xy={result.x[0]:.4f}")
    print(f"  Kd_xy={result.x[1]:.4f}")
    print(f"  Kp_z={result.x[2]:.4f}")
    print(f"  Kd_z={result.x[3]:.4f}")
    print(f"  Final error={result.fun:.4f}")

    return result.x

def optimize_ac():
    print("\nOptimizing Adaptive Controller for Cascaded System...")

    def objective(x):
        lambda_xy, lambda_z, k_xy, k_z, gamma_alpha, gamma_d = x
        controller = AdaptiveController(lambda_xy=lambda_xy, lambda_z=lambda_z,
                                       k_xy=k_xy, k_z=k_z,
                                       gamma_alpha=gamma_alpha, gamma_d=gamma_d)
        sim = spiral_opt(controller)
        try:
            sim.simulation()
            error = sim.get_error()
        except:
            error = 1000.0
        print(f"  λxy={lambda_xy:.2f} λz={lambda_z:.2f} kxy={k_xy:.2f} kz={k_z:.2f} " +
              f"γα={gamma_alpha:.2f} γd={gamma_d:.2f} -> error={error:.4f}")
        return error

    x0 = [10.0, 10.0, 15.0, 15.0, 0.1, 0.1]
    bounds = [(0.5, 20), (0.5, 20), (1, 30), (1, 30), (0.01, 2), (0.01, 2)]

    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                     options={'maxiter': 100, 'disp': True})

    print(f"\nOptimal AC Gains for Cascaded System:")
    print(f"  lambda_xy={result.x[0]:.4f}")
    print(f"  lambda_z={result.x[1]:.4f}")
    print(f"  k_xy={result.x[2]:.4f}")
    print(f"  k_z={result.x[3]:.4f}")
    print(f"  gamma_alpha={result.x[4]:.4f}")
    print(f"  gamma_d={result.x[5]:.4f}")
    print(f"  Final error={result.fun:.4f}")

    return result.x

if __name__ == "__main__":
    # Optimize both controllers
    print("=" * 60)
    print("OPTIMIZING CONTROLLERS FOR CASCADED SYSTEM")
    print("=" * 60)

    pd_gains = optimize_pd()
    ac_gains = optimize_ac()

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
