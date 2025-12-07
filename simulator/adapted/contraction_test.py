"""
Test simulator for ContractionAdaptiveQuadController.

This demonstrates the contraction-based adaptive control that adapts:
- Mass (via α = 1/m)
- 3D wind/disturbance forces (d_a)
"""

import numpy as np
import sys
sys.path.insert(0, '../../environment')
sys.path.insert(0, '../../controller')
sys.path.insert(0, '../../Drone')

from Quadcopter_Dynamics import environment
from contraction_adaptive_controller import ContractionAdaptiveQuadController
from Drone_1 import Drone_with_Package

import matplotlib.pyplot as plt

class simulator:
    def __init__(self):
        # Initialize drone with package (heavier)
        self.Drone = Drone_with_Package()

        # Initialize environment with heavier drone parameters
        self.env = environment(mass=self.Drone.m, Ixx=self.Drone.Ixx,
                              Iyy=self.Drone.Iyy, Izz=self.Drone.Izz)

        # Initialize contraction-based adaptive controller
        # Start with nominal mass (without package)
        self.controller = ContractionAdaptiveQuadController(
            g=9.81,
            # Position control gains (not used in contraction law, but kept for reference)
            Kp_xy=1.5, Kd_xy=1.0,
            Kp_z=5.0,  Kd_z=3.0,
            # Attitude control gains
            Kp_att=15.9895, Kd_att=2.0513,
            max_tilt_deg=35.0,
            # Contraction / sliding surface gains
            lambda_xy=2.0,
            lambda_z=2.0,
            k_xy=3.0,
            k_z=3.0,
            # Adaptation gains (tune these for performance)
            gamma_alpha=0.1,   # Mass adaptation gain
            gamma_dx=0.05,     # x-disturbance adaptation gain
            gamma_dy=0.05,     # y-disturbance adaptation gain
            gamma_dz=0.05,     # z-disturbance adaptation gain
            # Parameter bounds
            m_min=0.5,
            m_max=3.5,
            d_max=10.0
        )

        # Simulation parameters
        self.dt = 0.01
        self.max_time = 30.0
        self.max_time_steps = int(self.max_time / self.dt)

        # Initial state [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Data recording
        self.state_history = []
        self.control_history = []
        self.time_history = []
        self.time_step = 0.0

    def get_trajectory_point(self, t):
        """
        Define a complex 3D trajectory to test adaptation.

        Returns target position [x_d, y_d, z_d]
        """
        # Hover at different heights to test mass adaptation
        if t < 5.0:
            # Start: hover at 1m
            return np.array([0.0, 0.0, 1.0])
        elif t < 10.0:
            # Rise to 3m
            return np.array([0.0, 0.0, 3.0])
        elif t < 15.0:
            # Move in xy while at 3m (tests disturbance adaptation)
            return np.array([2.0, 2.0, 3.0])
        elif t < 20.0:
            # Circle in xy plane
            omega = 0.5  # rad/s
            radius = 1.5
            angle = omega * (t - 15.0)
            return np.array([2.0 + radius*np.cos(angle), 2.0 + radius*np.sin(angle), 3.0])
        else:
            # Return to origin and descend
            return np.array([0.0, 0.0, 1.5])

    def simulation(self):
        """Run the simulation loop."""
        print("Starting contraction-based adaptive simulation...")
        print(f"True mass with package: {self.Drone.m:.2f} kg")
        print(f"Nominal mass (controller starts with): {self.controller.m_nominal:.2f} kg")
        print(f"Mass mismatch: {abs(self.Drone.m - self.controller.m_nominal):.2f} kg\n")

        for step in range(self.max_time_steps):
            # Get current target position
            target = self.get_trajectory_point(self.time_step)

            # Compute control using contraction-based adaptive controller
            u = self.controller.controller(self.state, target, dt=self.dt)

            # Step environment dynamics
            state_dot = self.env.step(self.state, u)
            self.state = self.state + state_dot * self.dt

            # Record data every 10 steps
            if step % 10 == 0:
                self.state_history.append(self.state.copy())
                self.control_history.append(u.copy())
                self.time_history.append(self.time_step)
                self.controller.record_estimates(self.time_step)

            # Progress indicator
            if step % 100 == 0:
                progress = 100 * step / self.max_time_steps
                estimates = self.controller.get_estimates()
                print(f"Progress: {progress:.0f}% | t={self.time_step:.1f}s | "
                      f"m̂={estimates['m_hat']:.3f} kg | "
                      f"d̂=[{estimates['d_hat'][0]:.2f}, {estimates['d_hat'][1]:.2f}, {estimates['d_hat'][2]:.2f}] m/s²",
                      end='\r', flush=True)

            self.time_step += self.dt

        print("\nSimulation complete!")

        # Convert to numpy arrays
        time = np.array(self.time_history)
        states = np.array(self.state_history)
        controls = np.array(self.control_history)

        return time, states, controls

    def print_results(self):
        """Print final adaptation results."""
        estimates = self.controller.get_estimates()

        print("\n" + "="*60)
        print("FINAL ADAPTATION RESULTS")
        print("="*60)
        print(f"\nMass Estimation:")
        print(f"  True mass:      {estimates['m_true']:.3f} kg")
        print(f"  Estimated mass: {estimates['m_hat']:.3f} kg")
        print(f"  Error:          {estimates['m_error_pct']:.2f}%")

        print(f"\nDisturbance Estimation (d̂_a):")
        print(f"  d̂_x: {estimates['d_hat'][0]:.3f} m/s²")
        print(f"  d̂_y: {estimates['d_hat'][1]:.3f} m/s²")
        print(f"  d̂_z: {estimates['d_hat'][2]:.3f} m/s²")
        print("="*60 + "\n")

    def plot_results(self):
        """Plot trajectory tracking results."""
        if len(self.time_history) == 0:
            print("No data to plot!")
            return

        time = np.array(self.time_history)
        states = np.array(self.state_history)

        # Generate reference trajectory
        ref_traj = np.array([self.get_trajectory_point(t) for t in time])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Position tracking
        axes[0].plot(time, states[:, 0], 'b-', linewidth=2, label='x (actual)')
        axes[0].plot(time, ref_traj[:, 0], 'b--', linewidth=1.5, alpha=0.7, label='x (desired)')
        axes[0].plot(time, states[:, 1], 'g-', linewidth=2, label='y (actual)')
        axes[0].plot(time, ref_traj[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='y (desired)')
        axes[0].plot(time, states[:, 2], 'r-', linewidth=2, label='z (actual)')
        axes[0].plot(time, ref_traj[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='z (desired)')
        axes[0].set_xlabel('Time [s]', fontsize=12)
        axes[0].set_ylabel('Position [m]', fontsize=12)
        axes[0].set_title('Position Tracking (Contraction-Based Adaptive Control)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10, ncol=3)

        # Velocity
        axes[1].plot(time, states[:, 3], 'b-', linewidth=2, label='vx')
        axes[1].plot(time, states[:, 4], 'g-', linewidth=2, label='vy')
        axes[1].plot(time, states[:, 5], 'r-', linewidth=2, label='vz')
        axes[1].set_xlabel('Time [s]', fontsize=12)
        axes[1].set_ylabel('Velocity [m/s]', fontsize=12)
        axes[1].set_title('Velocity', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10, ncol=3)

        # Attitude
        axes[2].plot(time, np.degrees(states[:, 6]), 'b-', linewidth=2, label='φ (roll)')
        axes[2].plot(time, np.degrees(states[:, 7]), 'g-', linewidth=2, label='θ (pitch)')
        axes[2].plot(time, np.degrees(states[:, 8]), 'r-', linewidth=2, label='ψ (yaw)')
        axes[2].set_xlabel('Time [s]', fontsize=12)
        axes[2].set_ylabel('Angle [deg]', fontsize=12)
        axes[2].set_title('Attitude', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10, ncol=3)

        plt.tight_layout()
        plt.savefig('contraction_tracking.png', dpi=300, bbox_inches='tight')
        print("Tracking plot saved: contraction_tracking.png")
        plt.close(fig)


if __name__ == "__main__":
    print("="*60)
    print("CONTRACTION-BASED ADAPTIVE QUADROTOR CONTROL")
    print("Adapts: Mass (α=1/m) + 3D Disturbance (d_a)")
    print("="*60 + "\n")

    # Create and run simulation
    print("Initializing simulation...")
    sim = simulator()

    print(f"Running {sim.max_time}s simulation with dt={sim.dt}s...")
    time, states, controls = sim.simulation()

    print(f"\nSimulation completed: {len(time)} data points")

    # Display results
    sim.print_results()

    # Generate plots
    print("\nGenerating plots...")
    sim.plot_results()

    print("\n✅ Complete! Check the generated plot:")
    print("   - contraction_tracking.png (trajectory tracking)")
