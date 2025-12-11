import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))
from slalom import Slalom
from pd_controller import PDController
from ac_controller import AdaptiveController
from Drone import Drone
from scipy.optimize import minimize


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


def build_scenarios():
    scenarios = []
    # Baseline
    scenarios.append(("Baseline (m=1.9, no wind)", Drone, [0.0, 0.0, 0.0]))

    # Mass sweep
    mass_values = np.linspace(1.9, 4.0, 5)
    for m in mass_values:
        scenarios.append((f"Mass {m:.2f} kg, no wind", make_drone_class(m), [0.0, 0.0, 0.0]))

    # Wind sweep along [-2, 4, 0] scaled 0..10N
    wind_values = np.linspace(0.0, 10.0, 5)
    wind_dir = np.array([-2.0, 4.0, 0.0])
    norm_dir = wind_dir / np.linalg.norm(wind_dir)
    for w in wind_values:
        wvec = w * norm_dir if w != 0 else wind_dir * 0.0
        scenarios.append((f"Wind {wvec}", Drone, wvec.tolist()))

    # Combined mass + wind
    for m, w in zip(mass_values, wind_values):
        wvec = w * norm_dir if w != 0 else wind_dir * 0.0
        scenarios.append((f"Mass {m:.2f} kg + Wind {wvec}", make_drone_class(m), wvec.tolist()))

    return scenarios


def evaluate_controller(controller_ctor, scenarios, T=10.0):
    errors = []
    for _, drone_cls, wind in scenarios:
        sim = Slalom(controller_ctor(), drone_class=drone_cls, disturbance=wind)
        sim.T = T
        sim.run()
        mean_total, *_ = sim.errors()
        errors.append(mean_total)
    return float(np.mean(errors))


def optimize_pd():
    scenarios = build_scenarios()
    print("Optimizing PD over batch scenarios...")

    def objective(x):
        Kp_xy, Kd_xy, Kp_z, Kd_z, Ki_xy, Ki_z = x
        controller_ctor = lambda: PDController(Kp_xy=Kp_xy, Kd_xy=Kd_xy,
                                               Kp_z=Kp_z, Kd_z=Kd_z,
                                               Ki_xy=Ki_xy, Ki_z=Ki_z)
        cost = evaluate_controller(controller_ctor, scenarios, T=10.0)
        print(f"  Kp_xy={Kp_xy:.2f} Kd_xy={Kd_xy:.2f} Kp_z={Kp_z:.2f} Kd_z={Kd_z:.2f} Ki_xy={Ki_xy:.3f} Ki_z={Ki_z:.3f} -> cost={cost:.4f}")
        return cost

    x0 = [50.0, 12.5, 50.0, 11.5, 0.0001, 0.02]
    bounds = [(5, 60), (1, 20), (5, 60), (1, 20), (0, 0.5), (0, 0.5)]

    result = minimize(objective, x0, method='Powell', bounds=bounds,
                      options={'maxiter': 40, 'disp': True})

    print("\nOptimal PD Gains (batch):")
    print(f"  Kp_xy={result.x[0]:.4f}")
    print(f"  Kd_xy={result.x[1]:.4f}")
    print(f"  Kp_z={result.x[2]:.4f}")
    print(f"  Kd_z={result.x[3]:.4f}")
    print(f"  Ki_xy={result.x[4]:.4f}")
    print(f"  Ki_z={result.x[5]:.4f}")
    print(f"  Final cost={result.fun:.4f}")
    return result.x, result.fun


def optimize_ac():
    scenarios = build_scenarios()
    print("\nOptimizing AC over batch scenarios...")

    def objective(x):
        lambda_xy, lambda_z, k_xy, k_z, gamma_alpha, gamma_d = x
        controller_ctor = lambda: AdaptiveController(lambda_xy=lambda_xy, lambda_z=lambda_z,
                                                     k_xy=k_xy, k_z=k_z,
                                                     gamma_alpha=gamma_alpha, gamma_d=gamma_d)
        cost = evaluate_controller(controller_ctor, scenarios, T=10.0)
        print(f"  λxy={lambda_xy:.2f} λz={lambda_z:.2f} kxy={k_xy:.2f} kz={k_z:.2f} γα={gamma_alpha:.2f} γd={gamma_d:.2f} -> cost={cost:.4f}")
        return cost

    x0 = [3.6, 10.0, 15.0, 15.0, 0.3, 0.1]
    bounds = [(0.5, 12), (5, 15), (5, 25), (5, 25), (0.05, 5), (0.05, 1)]

    result = minimize(objective, x0, method='Powell', bounds=bounds,
                      options={'maxiter': 40, 'disp': True})

    print("\nOptimal AC Gains (batch):")
    print(f"  lambda_xy={result.x[0]:.4f}")
    print(f"  lambda_z={result.x[1]:.4f}")
    print(f"  k_xy={result.x[2]:.4f}")
    print(f"  k_z={result.x[3]:.4f}")
    print(f"  gamma_alpha={result.x[4]:.4f}")
    print(f"  gamma_d={result.x[5]:.4f}")
    print(f"  Final cost={result.fun:.4f}")
    return result.x, result.fun


if __name__ == "__main__":
    optimize_pd()
    optimize_ac()
