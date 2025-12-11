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
from scipy.optimize import minimize

class spiral_opt:
    def __init__(self, pos_controller, att_gains, drone_class=Drone, disturbance=None):
        self.state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.t = 0.0
        self.dt = 0.01
        self.r = 5
        self.env = CascadedQuadcopter(drone_class=drone_class, constant_disturbance=disturbance)
        self.pos_controller = pos_controller
        self.att_controller = AttitudeController(
            Kp_phi=att_gains[0],
            Kd_phi=att_gains[1],
            Kp_theta=att_gains[2],
            Kd_theta=att_gains[3],
            Kp_psi=att_gains[4],
            Kd_psi=att_gains[5]
        )
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
        self.step()
        full_state = self.env.step(self.u, self.dt)
        pos = full_state[0:3]
        vel = full_state[3:6]
        F_des = self.pos_controller.compute_control(pos, vel, self.state[:3], dt=self.dt)
        self.u = self.att_controller.compute_control(F_des, full_state)
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

def optimize_attitude_pd():
    print("Optimizing Attitude Controller with PD Position Controller...")

    def objective(x):
        Kp_phi, Kd_phi, Kp_theta, Kd_theta, Kp_psi, Kd_psi = x
        att_gains = [Kp_phi, Kd_phi, Kp_theta, Kd_theta, Kp_psi, Kd_psi]
        controller = PDController()
        sim = spiral_opt(controller, att_gains)
        sim.simulation()
        error = sim.get_error()
        print(f"  Kp_φ={Kp_phi:.2f} Kd_φ={Kd_phi:.2f} Kp_θ={Kp_theta:.2f} Kd_θ={Kd_theta:.2f} " +
              f"Kp_ψ={Kp_psi:.2f} Kd_ψ={Kd_psi:.2f} -> error={error:.4f}")
        return error

    x0 = [25.0, 5.0, 25.0, 5.0, 10.0, 2.0]
    bounds = [(5, 50), (1, 15), (5, 50), (1, 15), (1, 30), (0.5, 10)]

    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                     options={'maxiter': 50, 'disp': True})

    print(f"\nOptimal Attitude Gains (with PD):")
    print(f"  Kp_phi={result.x[0]:.4f}")
    print(f"  Kd_phi={result.x[1]:.4f}")
    print(f"  Kp_theta={result.x[2]:.4f}")
    print(f"  Kd_theta={result.x[3]:.4f}")
    print(f"  Kp_psi={result.x[4]:.4f}")
    print(f"  Kd_psi={result.x[5]:.4f}")
    print(f"  Final error={result.fun:.4f}")

    return result.x

def optimize_attitude_ac():
    print("\nOptimizing Attitude Controller with AC Position Controller...")

    def objective(x):
        Kp_phi, Kd_phi, Kp_theta, Kd_theta, Kp_psi, Kd_psi = x
        att_gains = [Kp_phi, Kd_phi, Kp_theta, Kd_theta, Kp_psi, Kd_psi]
        controller = AdaptiveController()
        sim = spiral_opt(controller, att_gains)
        sim.simulation()
        error = sim.get_error()
        print(f"  Kp_φ={Kp_phi:.2f} Kd_φ={Kd_phi:.2f} Kp_θ={Kp_theta:.2f} Kd_θ={Kd_theta:.2f} " +
              f"Kp_ψ={Kp_psi:.2f} Kd_ψ={Kd_psi:.2f} -> error={error:.4f}")
        return error

    x0 = [25.0, 5.0, 25.0, 5.0, 10.0, 2.0]
    bounds = [(5, 50), (1, 15), (5, 50), (1, 15), (1, 30), (0.5, 10)]

    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                     options={'maxiter': 50, 'disp': True})

    print(f"\nOptimal Attitude Gains (with AC):")
    print(f"  Kp_phi={result.x[0]:.4f}")
    print(f"  Kd_phi={result.x[1]:.4f}")
    print(f"  Kp_theta={result.x[2]:.4f}")
    print(f"  Kd_theta={result.x[3]:.4f}")
    print(f"  Kp_psi={result.x[4]:.4f}")
    print(f"  Kd_psi={result.x[5]:.4f}")
    print(f"  Final error={result.fun:.4f}")

    return result.x

if __name__ == "__main__":
    pd_att_gains = optimize_attitude_pd()
    ac_att_gains = optimize_attitude_ac()
