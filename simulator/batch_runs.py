import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))

from simple import spiral
from pd_controller import PDController
from ac_controller import AdaptiveController
from Drone import Drone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_drone_class(mass):
    """Factory to build a simple drone class with custom mass."""
    class CustomDrone:
        def __init__(self):
            self.Ixx = 0.028
            self.Iyy = 0.028
            self.Izz = 0.045
            self.m = mass
            self.F_max = 60
            self.tau_max = 4
    return CustomDrone


def run_scenario(controller, drone_class, disturbance):
    sim = spiral(controller, drone_class=drone_class, disturbance=disturbance)
    sim.simulation()
    ex, ey, ez, et, *_ = sim.get_errors()
    return et


def main():
    # Baseline (known parameters)
    scenarios = []
    scenarios.append(("Baseline (m=1.9, no wind)", Drone, [0.0, 0.0, 0.0]))

    # Mass sweep: 5 runs from 1.9 to 4.0 kg (no wind)
    mass_values = np.linspace(1.9, 4.0, 5)
    for m in mass_values:
        scenarios.append((f"Mass {m:.2f} kg, no wind", make_drone_class(m), [0.0, 0.0, 0.0]))

    # Wind sweep: 5 runs from 0 to 10 N (nominal mass)
    wind_values = np.linspace(0.0, 10.0, 5)
    for w in wind_values:
        scenarios.append((f"Wind [{w:.1f},0,0] N", Drone, [w, 0.0, 0.0]))

    # Combined mass + wind using the same value sets
    for m, w in zip(mass_values, wind_values):
        scenarios.append((f"Mass {m:.2f} kg + Wind [{w:.1f},0,0] N", make_drone_class(m), [w, 0.0, 0.0]))

    labels = []
    ac_errors = []
    pd_errors = []

    for idx, (label, drone_cls, wind) in enumerate(scenarios, start=1):
        print(f"Run {idx:02d}: {label}")
        ac_err = run_scenario(AdaptiveController(), drone_cls, wind)
        pd_err = run_scenario(PDController(), drone_cls, wind)
        labels.append(label)
        ac_errors.append(ac_err)
        pd_errors.append(pd_err)
        print(f"  AC total error: {ac_err:.3f}")
        print(f"  PD total error: {pd_err:.3f}\n")

    # Slice runs into categories
    base_ac = ac_errors[0]
    base_pd = pd_errors[0]

    mass_ac = ac_errors[1:6]
    mass_pd = pd_errors[1:6]

    wind_ac = ac_errors[6:11]
    wind_pd = pd_errors[6:11]

    comb_ac = ac_errors[11:16]
    comb_pd = pd_errors[11:16]

    x_base = np.arange(1, 6)
    x_mass = np.arange(1, len(mass_ac) + 1)
    x_wind = np.arange(1, len(wind_ac) + 1)
    x_comb = np.arange(1, len(comb_ac) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_base, np.full_like(x_base, base_ac, dtype=float), 'b--', label='AC baseline')
    plt.plot(x_base, np.full_like(x_base, base_pd, dtype=float), 'r--', label='PD baseline')

    plt.plot(x_mass, mass_ac, 'bo-', label='AC mass')
    plt.plot(x_mass, mass_pd, 'ro-', label='PD mass')

    plt.plot(x_wind, wind_ac, 'bs-', label='AC wind')
    plt.plot(x_wind, wind_pd, 'rs-', label='PD wind')

    plt.plot(x_comb, comb_ac, 'b^-', label='AC combined')
    plt.plot(x_comb, comb_pd, 'r^-', label='PD combined')

    plt.xlabel('Run index within category')
    plt.ylabel('Tracking error (mean total, m)')
    plt.title('Tracking Error Across Scenarios')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('batch_comparison_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
