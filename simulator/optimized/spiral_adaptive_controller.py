'''
3D spiral trajectory simulator with varying speed for quadcopter.
The drone follows an ascending spiral path with sinusoidally varying forward, angular, and climb speeds
to test controller performance on dynamic curved trajectories.
'''

import numpy as np
import sys
sys.path.insert(0, '../../environment')
sys.path.insert(0, '../../controller')
from Quadcopter_Dynamics import environment
from adaptive_controller import QuadcopterController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class simulator:
    def __init__(self):
        self.state = np.zeros(12)
        self.state[2] = 0.1
        self.dt = 0.01
        self.time_step = 0.0
        self.max_time = 50
        self.max_time_steps = int(self.max_time / self.dt)

        self.state_history = []
        self.control_history = []
        self.time_history = []
        self.error_history = []

        self.env = environment(mass=1.2, Ixx=0.028, Iyy=0.028, Izz=0.055)
        self.controller = QuadcopterController(mass=1.2, g=9.81)

    def get_target(self, t):
        # Base trajectory parameters
        radius = 3.0                # Radius of spiral (m)
        base_angular = 0.4          # Base angular frequency (rad/s)
        base_climb = 0.2            # Base climb rate (m/s)
        base_forward = 1.0          # Base forward speed (m/s)
        z_start = 1.0               # Starting altitude (m)
        speed_freq = 0.3            # Frequency of speed variation (rad/s)

        # Compute angular phase (integral of angular_freq(t))
        # angular_freq(t) = base_angular * (1 + 0.15 * sin(1.5 * speed_freq * t))
        # integral = base_angular * t - (0.15 * base_angular / (1.5 * speed_freq)) * cos(1.5 * speed_freq * t) + C
        ang_integral = base_angular * t - (0.15 * base_angular / (1.5 * speed_freq)) * (np.cos(1.5 * speed_freq * t) - 1.0)

        # Position with varying speeds (integrated from velocity functions)
        x_d = base_forward * (t - (0.2 / speed_freq) * (np.cos(speed_freq * t) - 1.0))
        y_d = radius * np.cos(ang_integral)
        z_d = z_start + base_climb * (t - (0.1 / (0.8 * speed_freq)) * (np.cos(0.8 * speed_freq * t) - 1.0))

        return np.array([x_d, y_d, z_d])

    def get_target_velocity(self, t):
        """Target velocity for 3D spiral with varying speed (derivative of position)"""
        # Base trajectory parameters (must match get_target())
        radius = 3.0
        base_angular = 0.4
        base_climb = 0.2
        base_forward = 1.0
        speed_freq = 0.3

        # Time-varying speeds
        forward_speed = base_forward * (1.0 + 0.2 * np.sin(speed_freq * t))
        angular_freq = base_angular * (1.0 + 0.15 * np.sin(1.5 * speed_freq * t))
        climb_rate = base_climb * (1.0 + 0.1 * np.sin(0.8 * speed_freq * t))

        # Compute angular phase for derivative calculation
        ang_integral = base_angular * t - (0.15 * base_angular / (1.5 * speed_freq)) * (np.cos(1.5 * speed_freq * t) - 1.0)

        # Velocity is time derivative of position
        vx_d = forward_speed
        vy_d = -radius * angular_freq * np.sin(ang_integral)
        vz_d = climb_rate

        return np.array([vx_d, vy_d, vz_d])

    def get_error(self, target):
        error_x = target[0] - self.state[0]
        error_y = target[1] - self.state[1]
        error_z = target[2] - self.state[2]
        return np.array([error_x, error_y, error_z])

    def print_results(self):
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)

        times = np.array(self.time_history)
        states = np.array(self.state_history)

        errors_x = []
        errors_y = []
        errors_z = []

        for i, t in enumerate(times):
            target = self.get_target(t)
            errors_x.append(target[0] - states[i, 0])
            errors_y.append(target[1] - states[i, 1])
            errors_z.append(target[2] - states[i, 2])

        errors_x = np.array(errors_x)
        errors_y = np.array(errors_y)
        errors_z = np.array(errors_z)
        total_error = np.sqrt(errors_x**2 + errors_y**2 + errors_z**2)

        print(f"\nError Statistics:")
        print(f"  X-axis:")
        print(f"    Mean:  {np.mean(errors_x):.4f} m")
        print(f"    RMS:   {np.sqrt(np.mean(errors_x**2)):.4f} m")
        print(f"    Max:   {np.max(np.abs(errors_x)):.4f} m")

        print(f"  Y-axis:")
        print(f"    Mean:  {np.mean(errors_y):.4f} m")
        print(f"    RMS:   {np.sqrt(np.mean(errors_y**2)):.4f} m")
        print(f"    Max:   {np.max(np.abs(errors_y)):.4f} m")

        print(f"  Z-axis:")
        print(f"    Mean:  {np.mean(errors_z):.4f} m")
        print(f"    RMS:   {np.sqrt(np.mean(errors_z**2)):.4f} m")
        print(f"    Max:   {np.max(np.abs(errors_z)):.4f} m")

        print(f"\n  Total Error:")
        print(f"    Mean:  {np.mean(total_error):.4f} m")
        print(f"    RMS:   {np.sqrt(np.mean(total_error**2)):.4f} m")
        print(f"    Max:   {np.max(total_error):.4f} m")

        print(f"\nFinal State:")
        print(f"  Position: ({states[-1,0]:.2f}, {states[-1,1]:.2f}, {states[-1,2]:.2f}) m")
        print(f"  Velocity: ({states[-1,3]:.2f}, {states[-1,4]:.2f}, {states[-1,5]:.2f}) m/s")

        final_target = self.get_target(times[-1])
        print(f"\nFinal Target:")
        print(f"  Position: ({final_target[0]:.2f}, {final_target[1]:.2f}, {final_target[2]:.2f}) m")

        print("="*60)

    def plot_results(self):
        times = np.array(self.time_history)
        states = np.array(self.state_history)
        controls = np.array(self.control_history)

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
        ax1.set_title('3D Spiral Trajectory with Varying Speed')
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
        ax3.plot(times, errors_x, 'r-', label='X error', linewidth=2)
        ax3.plot(times, errors_y, 'g-', label='Y error', linewidth=2)
        ax3.plot(times, errors_z, 'b-', label='Z error', linewidth=2)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error (m)')
        ax3.set_title('Component-wise Tracking Error')
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(3, 3, 4)
        ax4.plot(times, total_error, 'b-', linewidth=2)
        ax4.axhline(np.mean(total_error), color='r', linestyle='--',
                    label=f'Mean: {np.mean(total_error):.3f}m')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Total Error (m)')
        ax4.set_title('Total Tracking Error')
        ax4.legend()
        ax4.grid(True)

        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(times, states[:, 3], 'r-', label='vx', linewidth=2)
        ax5.plot(times, states[:, 4], 'g-', label='vy', linewidth=2)
        ax5.plot(times, states[:, 5], 'b-', label='vz', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Velocity (m/s)')
        ax5.set_title('Velocity Components')
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
        ax7.axhline(self.env.m * self.env.g, color='r', linestyle='--',
                    label=f'Hover thrust: {self.env.m * self.env.g:.1f}N', alpha=0.6)
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
        plt.savefig('spiral_results.png', dpi=300)
        print(f"\nPlot saved: spiral_results.png")
        plt.show()

    def simulation(self):
        for step in range(self.max_time_steps):
            target = self.get_target(self.time_step)
            target_vel = self.get_target_velocity(self.time_step)
            print(target)
            u = self.controller.controller(self.state, target, target_vel, dt=self.dt)
            print('u',u)
            state_dot = self.env.step(self.state, u)
            print('state_dot:',state_dot)
            self.state = self.state + state_dot * self.dt
            print('state',self.state)

            if step % 10 == 0:
                self.state_history.append(self.state.copy())
                self.control_history.append(u.copy())
                self.time_history.append(self.time_step)

            if step % 10 == 0:
                print(f"  Progress: {100*step/self.max_time_steps:.0f}% (step {step}/{self.max_time_steps})", end='\r', flush=True)

            self.time_step += self.dt

        print(f"  Progress: 100%")

        return np.array(self.time_history), np.array(self.state_history), np.array(self.control_history)


if __name__ == "__main__":
    print("Initializing simulation...")
    sim = simulator()

    print(f"Running {sim.max_time}s simulation with dt={sim.dt}s...")
    time, states, controls = sim.simulation()

    print(f"\nSimulation completed: {len(time)} data points")
    sim.print_results()
    sim.plot_results()
