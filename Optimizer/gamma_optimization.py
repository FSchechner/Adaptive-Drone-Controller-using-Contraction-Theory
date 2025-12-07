import numpy as np
import sys
import os
sys.path.insert(0, '../environment')
sys.path.insert(0, '../controller')
sys.path.insert(0,'../Drone')

from Quadcopter_Dynamics import environment
from contraction_adaptive_controller import ContractionAdaptiveQuadController
from Drone_1 import Drone_with_Package
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GammaOptimizer:
    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations
        self.iteration = 0
        self.best_cost = float('inf')
        self.best_params = None

        self.dt = 0.01
        self.max_time = 30
        self.max_time_steps = int(self.max_time / self.dt)

        self.w_tracking = 1.0
        self.w_convergence = 2.0
        self.w_oscillation = 0.5

    def get_target(self, t):
        if t < 6.0:
            gate_width = 2.5
            slalom_freq = 0.5
            x_d = 2.0 * t
            y_d = gate_width * np.sin(slalom_freq * np.pi * t)
            z_d = 2.0 + 0.2 * t
        elif t < 12.0:
            t_local = t - 6.0
            radius = 3.0
            omega = 0.6
            x_d = 12.0 + radius * np.cos(omega * t_local)
            y_d = radius * np.sin(omega * t_local)
            z_d = 3.2 + 0.3 * t_local
        elif t < 18.0:
            t_local = t - 12.0
            omega = 0.5
            radius_8 = 4.0
            x_d = 15.0 + radius_8 * np.sin(omega * t_local)
            y_d = 2.0 * np.sin(2 * omega * t_local)
            z_d = 5.0
        elif t < 24.0:
            t_local = t - 18.0
            x_d = 15.0 + 2.5 * t_local + (0.5/0.8) * (1 - np.cos(0.8 * t_local))
            y_d = 1.5 * np.sin(0.6 * t_local)
            z_d = 5.0 + 0.8 * np.sin(t_local)
        else:
            t_local = t - 24.0
            decay = np.exp(-0.5 * t_local)
            x_d = 30.0 + 3.0 * decay
            y_d = 2.0 * decay * np.sin(0.6 * 24.0)
            z_d = 5.0 - 2.0 * (1 - decay)
        return np.array([x_d, y_d, z_d])

    def run_simulation(self, params):
        try:
            gamma_alpha, gamma_dx, gamma_dy, gamma_dz = params

            drone = Drone_with_Package()
            env = environment(mass=drone.m, Ixx=drone.Ixx, Iyy=drone.Iyy, Izz=drone.Izz)
            controller = ContractionAdaptiveQuadController(
                gamma_alpha=gamma_alpha,
                gamma_dx=gamma_dx,
                gamma_dy=gamma_dy,
                gamma_dz=gamma_dz
            )

            state = np.zeros(12)
            state[2] = 0.1

            tracking_errors = []
            mass_errors = []
            param_derivatives = []

            time_step = 0.0
            theta_prev = controller.theta_hat.copy()
            for step in range(self.max_time_steps):
                target = self.get_target(time_step)
                u = controller.controller(state, target, dt=self.dt)
                state_dot = env.step(state, u)
                state = state + state_dot * self.dt

                error = target - state[0:3]
                tracking_error = np.linalg.norm(error)
                tracking_errors.append(tracking_error)

                m_error = abs(controller.theta_hat[0] - 1.0/drone.m) / (1.0/drone.m)
                mass_errors.append(m_error)

                if step > 0:
                    theta_change = controller.theta_hat - theta_prev
                    param_derivative = np.linalg.norm(theta_change) / self.dt
                    param_derivatives.append(param_derivative)
                theta_prev = controller.theta_hat.copy()

                time_step += self.dt

                if np.any(np.isnan(state)) or np.any(np.abs(state) > 1e6):
                    return 1e10

            rms_tracking = np.sqrt(np.mean(np.array(tracking_errors)**2))

            final_mass_error = mass_errors[-1]
            convergence_cost = final_mass_error

            mean_param_derivative = np.mean(param_derivatives) if param_derivatives else 0.0

            cost = (self.w_tracking * rms_tracking +
                    self.w_convergence * convergence_cost +
                    self.w_oscillation * mean_param_derivative)

            return cost

        except Exception as e:
            return 1e10

    def objective_function(self, params):
        self.iteration += 1
        cost = self.run_simulation(params)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            print(f"Iteration {self.iteration}: New best cost = {cost:.6f}")
            print(f"  gamma_alpha={params[0]:.4f}, gamma_dx={params[1]:.4f}, gamma_dy={params[2]:.4f}, gamma_dz={params[3]:.4f}")
        elif self.iteration % 5 == 0:
            print(f"Iteration {self.iteration}: cost = {cost:.6f}")

        return cost

    def optimize(self):
        print("="*70)
        print("GAMMA PARAMETER OPTIMIZATION FOR CONTRACTION ADAPTIVE CONTROLLER")
        print("="*70)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Simulation time: {self.max_time}s")
        print(f"Tracking weight: {self.w_tracking}")
        print(f"Convergence weight: {self.w_convergence}")
        print(f"Oscillation weight: {self.w_oscillation}")
        print("="*70)

        initial_params = np.array([0.5, 0.5, 0.5, 0.5])

        bounds = [
            (0.01, 5.0),   # gamma_alpha
            (0.01, 5.0),   # gamma_dx
            (0.01, 5.0),   # gamma_dy
            (0.01, 5.0)    # gamma_dz
        ]

        print("\nInitial parameters:")
        print(f"  gamma_alpha={initial_params[0]:.4f}")
        print(f"  gamma_dx={initial_params[1]:.4f}")
        print(f"  gamma_dy={initial_params[2]:.4f}")
        print(f"  gamma_dz={initial_params[3]:.4f}")

        initial_cost = self.run_simulation(initial_params)
        print(f"\nInitial cost: {initial_cost:.6f}")
        print("\nStarting optimization...\n")

        from scipy.optimize import differential_evolution

        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=self.max_iterations,
            popsize=8,
            seed=42,
            disp=False
        )

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Total iterations: {self.iteration}")
        print(f"\nBest cost: {self.best_cost:.6f}")
        print("\nOptimal parameters:")
        print(f"  gamma_alpha = {self.best_params[0]:.4f}")
        print(f"  gamma_dx = {self.best_params[1]:.4f}")
        print(f"  gamma_dy = {self.best_params[2]:.4f}")
        print(f"  gamma_dz = {self.best_params[3]:.4f}")
        print(f"\nImprovement: {100 * (initial_cost - self.best_cost) / initial_cost:.2f}%")
        print("="*70)

        with open('optimal_gamma.txt', 'w') as f:
            f.write("Optimal Gamma Parameters for Contraction Adaptive Controller\n")
            f.write("="*60 + "\n\n")
            f.write(f"Initial cost: {initial_cost:.6f}\n")
            f.write(f"Final cost: {self.best_cost:.6f}\n")
            f.write(f"Improvement: {100 * (initial_cost - self.best_cost) / initial_cost:.2f}%\n\n")
            f.write("Parameters:\n")
            f.write(f"  gamma_alpha = {self.best_params[0]:.4f}\n")
            f.write(f"  gamma_dx = {self.best_params[1]:.4f}\n")
            f.write(f"  gamma_dy = {self.best_params[2]:.4f}\n")
            f.write(f"  gamma_dz = {self.best_params[3]:.4f}\n")
            f.write("\nTo use in ContractionAdaptiveQuadController:\n")
            f.write(f"controller = ContractionAdaptiveQuadController(\n")
            f.write(f"    gamma_alpha={self.best_params[0]:.4f},\n")
            f.write(f"    gamma_dx={self.best_params[1]:.4f},\n")
            f.write(f"    gamma_dy={self.best_params[2]:.4f},\n")
            f.write(f"    gamma_dz={self.best_params[3]:.4f}\n")
            f.write(f")\n")

        print("\nResults saved to: optimal_gamma.txt")

        print("\nGenerating plots for best parameters...")
        self.plot_best_result()

        return self.best_params, self.best_cost

    def plot_best_result(self):
        gamma_alpha, gamma_dx, gamma_dy, gamma_dz = self.best_params

        drone = Drone_with_Package()
        env = environment(mass=drone.m, Ixx=drone.Ixx, Iyy=drone.Iyy, Izz=drone.Izz)
        controller = ContractionAdaptiveQuadController(
            gamma_alpha=gamma_alpha,
            gamma_dx=gamma_dx,
            gamma_dy=gamma_dy,
            gamma_dz=gamma_dz
        )

        state = np.zeros(12)
        state[2] = 0.1

        times = []
        states = []
        controls = []

        time_step = 0.0
        for step in range(self.max_time_steps):
            target = self.get_target(time_step)
            u = controller.controller(state, target, dt=self.dt)
            state_dot = env.step(state, u)
            state = state + state_dot * self.dt

            if step % 10 == 0:
                times.append(time_step)
                states.append(state.copy())
                controls.append(u.copy())
                controller.record_estimates(time_step)

            time_step += self.dt

        times = np.array(times)
        states = np.array(states)
        controls = np.array(controls)

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

        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax1.plot(targets_x, targets_y, targets_z, 'b--', linewidth=2, label='Target', alpha=0.6)
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2, label='Actual')
        ax1.scatter([states[0, 0]], [states[0, 1]], [states[0, 2]], c='g', s=100, marker='o', label='Start')
        ax1.scatter([states[-1, 0]], [states[-1, 1]], [states[-1, 2]], c='r', s=100, marker='x', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory - Optimized Gamma')
        ax1.legend()
        ax1.grid(True)

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

        ax3 = fig.add_subplot(3, 3, 3)
        ax3.plot(times, total_error, 'b-', linewidth=2)
        ax3.axhline(np.mean(total_error), color='r', linestyle='--',
                    label=f'Mean: {np.mean(total_error):.3f}m')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Total Error (m)')
        ax3.set_title('Total Tracking Error')
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(3, 3, 4)
        m_hist = np.array(controller.estimate_history['m_hat'])
        ax4.plot(times, m_hist, 'b-', linewidth=2, label='Estimated')
        ax4.axhline(y=drone.m, color='r', linestyle='--', linewidth=2, label=f'True: {drone.m:.2f} kg')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Mass (kg)')
        ax4.set_title('Mass Adaptation')
        ax4.legend()
        ax4.grid(True)

        ax5 = fig.add_subplot(3, 3, 5)
        d_x_hist = np.array(controller.estimate_history['d_hat_x'])
        d_y_hist = np.array(controller.estimate_history['d_hat_y'])
        d_z_hist = np.array(controller.estimate_history['d_hat_z'])
        ax5.plot(times, d_x_hist, 'r-', linewidth=2, label='d_x')
        ax5.plot(times, d_y_hist, 'g-', linewidth=2, label='d_y')
        ax5.plot(times, d_z_hist, 'b-', linewidth=2, label='d_z')
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Disturbance (m/s²)')
        ax5.set_title('Disturbance Adaptation')
        ax5.legend()
        ax5.grid(True)

        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(times, np.degrees(states[:, 6]), 'r-', label='φ (roll)', linewidth=2)
        ax6.plot(times, np.degrees(states[:, 7]), 'g-', label='θ (pitch)', linewidth=2)
        ax6.plot(times, np.degrees(states[:, 8]), 'b-', label='ψ (yaw)', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angle (deg)')
        ax6.set_title('Euler Angles')
        ax6.legend()
        ax6.grid(True)

        ax7 = fig.add_subplot(3, 3, 7)
        ax7.plot(times, controls[:, 0], 'b-', linewidth=2)
        ax7.axhline(env.m * env.g, color='r', linestyle='--',
                    label=f'Hover: {env.m * env.g:.1f}N', alpha=0.6)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Thrust (N)')
        ax7.set_title('Thrust Command')
        ax7.legend()
        ax7.grid(True)

        ax8 = fig.add_subplot(3, 3, 8)
        ax8.plot(times, controls[:, 1], 'r-', label='τφ', linewidth=2)
        ax8.plot(times, controls[:, 2], 'g-', label='τθ', linewidth=2)
        ax8.plot(times, controls[:, 3], 'b-', label='τψ', linewidth=2)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Torque (Nm)')
        ax8.set_title('Torque Commands')
        ax8.legend()
        ax8.grid(True)

        ax9 = fig.add_subplot(3, 3, 9)
        speed = np.sqrt(states[:, 3]**2 + states[:, 4]**2 + states[:, 5]**2)
        ax9.plot(times, speed, 'b-', linewidth=2)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Speed (m/s)')
        ax9.set_title('Total Speed')
        ax9.grid(True)

        plt.tight_layout()
        plt.savefig('gamma_optimization_result.png', dpi=300)
        print(f"Plot saved: gamma_optimization_result.png")
        plt.close()


if __name__ == "__main__":
    optimizer = GammaOptimizer(max_iterations=50)
    optimal_params, optimal_cost = optimizer.optimize()
