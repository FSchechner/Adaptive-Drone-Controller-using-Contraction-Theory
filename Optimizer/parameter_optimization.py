import numpy as np
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '../simulator')
sys.path.insert(0, '../environment')
sys.path.insert(0, '../controller')

from Quadcopter_Dynamics import environment
from adaptive_controller import QuadcopterController


class ParameterOptimizer:
    def __init__(self, max_iterations=100, tau_max=3.5):
        self.max_iterations = max_iterations
        self.iteration = 0
        self.best_cost = float('inf')
        self.best_params = None

        # Simulation parameters
        self.dt = 0.01
        self.max_time = 25
        self.max_time_steps = int(self.max_time / self.dt)

        # Controller constraints
        self.tau_max = tau_max  # Torque limit (must match controller)

        # Weights for objective function
        self.w_tracking = 1.0      # Weight for tracking error
        self.w_control = 0.01      # Weight for control effort
        self.w_oscillation = 0.4   # Weight for torque derivative (oscillations)

    def get_target(self, t):
        """Target trajectory - straight line"""
        speed = 2.0
        altitude = 3.0
        x_d = speed * t
        y_d = 0.0
        z_d = altitude
        return np.array([x_d, y_d, z_d])

    def get_target_velocity(self, t):
        """Target velocity for straight line trajectory"""
        speed = 2.0
        vx_d = speed
        vy_d = 0.0
        vz_d = 0.0
        return np.array([vx_d, vy_d, vz_d])

    def run_simulation(self, params):
        """Run simulation with given parameters and return cost"""
        try:
            # Unpack parameters
            Kp_long, Kd_long, Kp_lat, Kd_lat, Kp_z, Kd_z, Kp_att, Kd_att = params

            # Create environment and controller with these parameters
            env = environment(mass=1.2, Ixx=0.028, Iyy=0.028, Izz=0.055)
            controller = QuadcopterController(
                mass=1.2, g=9.81,
                Kp_long=Kp_long, Kd_long=Kd_long,
                Kp_lat=Kp_lat, Kd_lat=Kd_lat,
                Kp_z=Kp_z, Kd_z=Kd_z,
                Kp_att=Kp_att, Kd_att=Kd_att,
                max_tilt_deg=35.0,
                F_max=25.0,
                tau_max=self.tau_max
            )

            # Initialize state
            state = np.zeros(12)
            state[2] = 0.1  # Start at 0.1m altitude

            # Storage
            tracking_errors = []
            control_efforts = []
            torque_derivatives = []

            # Run simulation
            time_step = 0.0
            u_prev = None
            for step in range(self.max_time_steps):
                target = self.get_target(time_step)
                target_vel = self.get_target_velocity(time_step)
                u = controller.controller(state, target, target_vel)
                state_dot = env.step(state, u)
                state = state + state_dot * self.dt

                # Calculate tracking error
                error = target - state[0:3]
                tracking_error = np.linalg.norm(error)
                tracking_errors.append(tracking_error)

                # Calculate control effort (normalize thrust and torques)
                thrust_effort = (u[0] / 25.0)**2  # Normalize by max thrust
                torque_effort = np.sum((u[1:4] / self.tau_max)**2)  # Normalize by max torque
                control_effort = thrust_effort + torque_effort
                control_efforts.append(control_effort)

                # Calculate torque derivative (oscillation penalty)
                if u_prev is not None:
                    torque_change = u[1:4] - u_prev[1:4]  # Change in torques
                    torque_derivative = np.linalg.norm(torque_change) / self.dt
                    torque_derivatives.append(torque_derivative)
                u_prev = u.copy()

                time_step += self.dt

                # Check for NaN or instability
                if np.any(np.isnan(state)) or np.any(np.abs(state) > 1e6):
                    return 1e10  # Return huge cost for unstable solutions

            # Calculate total cost
            mean_tracking_error = np.mean(tracking_errors)
            rms_tracking_error = np.sqrt(np.mean(np.array(tracking_errors)**2))
            mean_control_effort = np.mean(control_efforts)
            mean_torque_derivative = np.mean(torque_derivatives) if torque_derivatives else 0.0

            cost = (self.w_tracking * rms_tracking_error +
                    self.w_control * mean_control_effort +
                    self.w_oscillation * mean_torque_derivative)

            return cost

        except Exception as e:
            # Return huge cost if simulation fails
            return 1e10

    def objective_function(self, params):
        """Objective function to minimize"""
        self.iteration += 1

        cost = self.run_simulation(params)

        # Track best solution
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            print(f"Iteration {self.iteration}: New best cost = {cost:.6f}")
            print(f"  Kp_long={params[0]:.3f}, Kd_long={params[1]:.3f}")
            print(f"  Kp_lat={params[2]:.3f}, Kd_lat={params[3]:.3f}")
            print(f"  Kp_z={params[4]:.3f}, Kd_z={params[5]:.3f}")
            print(f"  Kp_att={params[6]:.3f}, Kd_att={params[7]:.3f}")
        elif self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: cost = {cost:.6f}")

        return cost

    def optimize(self):
        """Run optimization"""
        print("="*70)
        print("QUADCOPTER PARAMETER OPTIMIZATION")
        print("="*70)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Simulation time: {self.max_time}s")
        print(f"Torque limit: {self.tau_max} Nm")
        print(f"Tracking weight: {self.w_tracking}")
        print(f"Control weight: {self.w_control}")
        print(f"Oscillation weight: {self.w_oscillation}")
        print("="*70)

        # Initial parameters (current defaults)
        initial_params = np.array([
            1.0,   # Kp_long
            0.8,   # Kd_long
            0.6,   # Kp_lat
            0.6,   # Kd_lat
            5.0,   # Kp_z
            3.0,   # Kd_z
            15.0,  # Kp_att
            5.0    # Kd_att
        ])

        # Parameter bounds
        bounds = [
            (0.1, 6),    # Kp_long
            (0.1, 6.0),    # Kd_long
            (0.1, 6.0),    # Kp_lat
            (0.1, 6.0),    # Kd_lat
            (1.0, 15.0),   # Kp_z
            (0.5, 8.0),    # Kd_z
            (5.0, 30.0),   # Kp_att
            (1.0, 15.0)    # Kd_att
        ]

        print("\nInitial parameters:")
        print(f"  Kp_long={initial_params[0]:.3f}, Kd_long={initial_params[1]:.3f}")
        print(f"  Kp_lat={initial_params[2]:.3f}, Kd_lat={initial_params[3]:.3f}")
        print(f"  Kp_z={initial_params[4]:.3f}, Kd_z={initial_params[5]:.3f}")
        print(f"  Kp_att={initial_params[6]:.3f}, Kd_att={initial_params[7]:.3f}")

        initial_cost = self.run_simulation(initial_params)
        print(f"\nInitial cost: {initial_cost:.6f}")
        print("\nStarting optimization...\n")

        # Use scipy.optimize
        from scipy.optimize import minimize, differential_evolution

        # Method 1: Nelder-Mead (local optimization)
        if self.max_iterations <= 100:
            result = minimize(
                self.objective_function,
                initial_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': self.max_iterations, 'disp': False}
            )
            optimal_params = result.x
        else:
            # Method 2: Differential Evolution (global optimization) for more iterations
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=self.max_iterations // 10,  # Each iteration tests population
                popsize=10,
                seed=42,
                disp=False
            )
            optimal_params = result.x

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Total iterations: {self.iteration}")
        print(f"\nBest cost: {self.best_cost:.6f}")
        print("\nOptimal parameters:")
        print(f"  Kp_long = {self.best_params[0]:.4f}")
        print(f"  Kd_long = {self.best_params[1]:.4f}")
        print(f"  Kp_lat  = {self.best_params[2]:.4f}")
        print(f"  Kd_lat  = {self.best_params[3]:.4f}")
        print(f"  Kp_z    = {self.best_params[4]:.4f}")
        print(f"  Kd_z    = {self.best_params[5]:.4f}")
        print(f"  Kp_att  = {self.best_params[6]:.4f}")
        print(f"  Kd_att  = {self.best_params[7]:.4f}")

        print("\nImprovement: {:.2f}%".format(100 * (initial_cost - self.best_cost) / initial_cost))
        print("="*70)

        # Save results to file
        with open('optimal_parameters.txt', 'w') as f:
            f.write("Optimal Quadcopter Controller Parameters\n")
            f.write("="*50 + "\n\n")
            f.write(f"Torque limit: {self.tau_max} Nm\n")
            f.write(f"Initial cost: {initial_cost:.6f}\n")
            f.write(f"Final cost: {self.best_cost:.6f}\n")
            f.write(f"Improvement: {100 * (initial_cost - self.best_cost) / initial_cost:.2f}%\n\n")
            f.write("Parameters:\n")
            f.write(f"  Kp_long = {self.best_params[0]:.4f}\n")
            f.write(f"  Kd_long = {self.best_params[1]:.4f}\n")
            f.write(f"  Kp_lat  = {self.best_params[2]:.4f}\n")
            f.write(f"  Kd_lat  = {self.best_params[3]:.4f}\n")
            f.write(f"  Kp_z    = {self.best_params[4]:.4f}\n")
            f.write(f"  Kd_z    = {self.best_params[5]:.4f}\n")
            f.write(f"  Kp_att  = {self.best_params[6]:.4f}\n")
            f.write(f"  Kd_att  = {self.best_params[7]:.4f}\n")
            f.write("\nTo use these parameters in QuadcopterController:\n")
            f.write(f"controller = QuadcopterController(\n")
            f.write(f"    mass=1.2, g=9.81,\n")
            f.write(f"    Kp_long={self.best_params[0]:.4f}, Kd_long={self.best_params[1]:.4f},\n")
            f.write(f"    Kp_lat={self.best_params[2]:.4f}, Kd_lat={self.best_params[3]:.4f},\n")
            f.write(f"    Kp_z={self.best_params[4]:.4f}, Kd_z={self.best_params[5]:.4f},\n")
            f.write(f"    Kp_att={self.best_params[6]:.4f}, Kd_att={self.best_params[7]:.4f},\n")
            f.write(f"    max_tilt_deg=35.0, F_max=25.0, tau_max={self.tau_max}\n")
            f.write(f")\n")

        print("\nResults saved to: optimal_parameters.txt")

        # Plot the best result
        print("\nGenerating plots for best parameters...")
        self.plot_best_result()

        return self.best_params, self.best_cost

    def plot_best_result(self):
        """Run simulation with best parameters and generate plots"""
        # Run simulation with best parameters
        params = self.best_params
        Kp_long, Kd_long, Kp_lat, Kd_lat, Kp_z, Kd_z, Kp_att, Kd_att = params

        env = environment(mass=1.2, Ixx=0.028, Iyy=0.028, Izz=0.055)
        controller = QuadcopterController(
            mass=1.2, g=9.81,
            Kp_long=Kp_long, Kd_long=Kd_long,
            Kp_lat=Kp_lat, Kd_lat=Kd_lat,
            Kp_z=Kp_z, Kd_z=Kd_z,
            Kp_att=Kp_att, Kd_att=Kd_att,
            max_tilt_deg=35.0,
            F_max=25.0,
            tau_max=self.tau_max
        )

        # Initialize state
        state = np.zeros(12)
        state[2] = 0.1

        # Storage
        times = []
        states = []
        controls = []

        # Run simulation
        time_step = 0.0
        for step in range(self.max_time_steps):
            target = self.get_target(time_step)
            target_vel = self.get_target_velocity(time_step)
            u = controller.controller(state, target, target_vel)
            state_dot = env.step(state, u)
            state = state + state_dot * self.dt

            if step % 10 == 0:
                times.append(time_step)
                states.append(state.copy())
                controls.append(u.copy())

            time_step += self.dt

        times = np.array(times)
        states = np.array(states)
        controls = np.array(controls)

        # Calculate targets and errors
        targets_x = []
        targets_y = []
        targets_z = []
        errors_x = []
        errors_y = []
        errors_z = []

        for i, t in enumerate(times):
            target = self.get_target(t)
            targets_x.append(target[0])
            targets_y.append(target[1])
            targets_z.append(target[2])
            errors_x.append(target[0] - states[i, 0])
            errors_y.append(target[1] - states[i, 1])
            errors_z.append(target[2] - states[i, 2])

        targets_x = np.array(targets_x)
        targets_y = np.array(targets_y)
        targets_z = np.array(targets_z)
        errors_x = np.array(errors_x)
        errors_y = np.array(errors_y)
        errors_z = np.array(errors_z)
        total_error = np.sqrt(errors_x**2 + errors_y**2 + errors_z**2)

        # Create plots
        fig = plt.figure(figsize=(16, 12))

        # 3D Trajectory
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax1.plot(targets_x, targets_y, targets_z, 'b--', linewidth=2, label='Target', alpha=0.6)
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2, label='Actual')
        ax1.scatter([states[0, 0]], [states[0, 1]], [states[0, 2]], c='g', s=100, marker='o', label='Start')
        ax1.scatter([states[-1, 0]], [states[-1, 1]], [states[-1, 2]], c='r', s=100, marker='x', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory - Optimized Parameters')
        ax1.legend()
        ax1.grid(True)

        # Position vs Time
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.plot(times, states[:, 0], 'r-', label='Actual X', linewidth=2)
        ax2.plot(times, targets_x, 'r--', label='Target X', linewidth=2, alpha=0.6)
        ax2.plot(times, states[:, 1], 'g-', label='Actual Y', linewidth=2)
        ax2.plot(times, targets_y, 'g--', label='Target Y', linewidth=2, alpha=0.6)
        ax2.plot(times, states[:, 2], 'b-', label='Actual Z', linewidth=2)
        ax2.plot(times, targets_z, 'b--', label='Target Z', linewidth=2, alpha=0.6)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True)

        # Component-wise Error
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.plot(times, errors_x, 'r-', label='X error', linewidth=2)
        ax3.plot(times, errors_y, 'g-', label='Y error', linewidth=2)
        ax3.plot(times, errors_z, 'b-', label='Z error', linewidth=2)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error (m)')
        ax3.set_title('Component-wise Tracking Error')
        ax3.legend()
        ax3.grid(True)

        # Total Error
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.plot(times, total_error, 'b-', linewidth=2)
        ax4.axhline(np.mean(total_error), color='r', linestyle='--',
                    label=f'Mean: {np.mean(total_error):.3f}m')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Total Error (m)')
        ax4.set_title('Total Tracking Error')
        ax4.legend()
        ax4.grid(True)

        # Velocity
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(times, states[:, 3], 'r-', label='vx', linewidth=2)
        ax5.plot(times, states[:, 4], 'g-', label='vy', linewidth=2)
        ax5.plot(times, states[:, 5], 'b-', label='vz', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Velocity (m/s)')
        ax5.set_title('Velocity Components')
        ax5.legend()
        ax5.grid(True)

        # Euler Angles
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(times, np.degrees(states[:, 6]), 'r-', label='φ (roll)', linewidth=2)
        ax6.plot(times, np.degrees(states[:, 7]), 'g-', label='θ (pitch)', linewidth=2)
        ax6.plot(times, np.degrees(states[:, 8]), 'b-', label='ψ (yaw)', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angle (deg)')
        ax6.set_title('Euler Angles')
        ax6.legend()
        ax6.grid(True)

        # Thrust
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.plot(times, controls[:, 0], 'b-', linewidth=2)
        ax7.axhline(env.m * env.g, color='r', linestyle='--',
                    label=f'Hover thrust: {env.m * env.g:.1f}N', alpha=0.6)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Thrust (N)')
        ax7.set_title('Thrust Command')
        ax7.legend()
        ax7.grid(True)

        # Torques
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.plot(times, controls[:, 1], 'r-', label='τφ', linewidth=2)
        ax8.plot(times, controls[:, 2], 'g-', label='τθ', linewidth=2)
        ax8.plot(times, controls[:, 3], 'b-', label='τψ', linewidth=2)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Torque (Nm)')
        ax8.set_title('Torque Commands')
        ax8.legend()
        ax8.grid(True)

        # Total Speed
        ax9 = fig.add_subplot(3, 3, 9)
        speed = np.sqrt(states[:, 3]**2 + states[:, 4]**2 + states[:, 5]**2)
        ax9.plot(times, speed, 'b-', linewidth=2)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Speed (m/s)')
        ax9.set_title('Total Speed')
        ax9.grid(True)

        plt.tight_layout()
        plt.savefig('optimization_result.png', dpi=300)
        print(f"Plot saved: optimization_result.png")
        plt.show()


if __name__ == "__main__":
    # tau_max should match the controller's default value
    optimizer = ParameterOptimizer(max_iterations=100, tau_max=3.5)
    optimal_params, optimal_cost = optimizer.optimize()
