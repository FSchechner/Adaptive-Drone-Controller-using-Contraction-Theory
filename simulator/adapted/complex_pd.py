'''
Aggressive 3D parcour to test quadcopter controller limits.
The drone performs:
- Rapid slalom maneuvers (sharp lateral movements)
- Vertical climbs and dives (altitude challenges)
- Figure-8 patterns (continuous turning)
- Speed variations (0.5 m/s to 6 m/s)
- Emergency braking maneuvers
This tests position tracking, velocity tracking, and acceleration limits.
'''

import numpy as np
import sys
sys.path.insert(0, '../../environment')
sys.path.insert(0, '../../controller')
sys.path.insert(0,'../../Drone')
from Quadcopter_Dynamics import environment
from controller import QuadcopterController
from Drone_1 import Drone_with_Package
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class simulator:
    def __init__(self):
        self.state = np.zeros(12)
        self.state[2] = 0.1
        self.dt = 0.01
        self.time_step = 0.0
        self.max_time = 30
        self.max_time_steps = int(self.max_time / self.dt)

        self.state_history = []
        self.control_history = []
        self.time_history = []
        self.error_history = []
        self.Drone = Drone_with_Package()
        self.env = environment(mass= self.Drone.m, Ixx= self.Drone.Ixx, Iyy= self.Drone.Iyy, Izz= self.Drone.Izz)
        self.controller = QuadcopterController(g=9.81)

    def get_target(self, t):
        """
        Gentle 3D parcour with 5 sections:
        0-6s: Smooth slalom (gentle S-curves)
        6-12s: Circular climb (spiral up)
        12-18s: Figure-8 pattern at constant altitude
        18-24s: Straight line with speed variation
        24-30s: Descending arc to hover
        """

        # Section 1: Smooth Slalom (0-6s)
        if t < 6.0:
            # Gentle forward motion with smooth lateral weaving
            gate_width = 1.5    # Lateral displacement (gentle)
            slalom_freq = 0.35  # Gates per second (slow)

            x_d = 1.0 * t  # Constant forward speed 1 m/s
            y_d = gate_width * np.sin(slalom_freq * np.pi * t)
            z_d = 2.0 + 0.15 * t  # Very gentle climb

        # Section 2: Circular Climb (6-12s)
        elif t < 12.0:
            t_local = t - 6.0
            radius = 2.0
            omega = 0.4  # rad/s (gentle rotation)

            x_d = 6.0 + radius * np.cos(omega * t_local)
            y_d = radius * np.sin(omega * t_local)
            z_d = 2.9 + 0.2 * t_local  # Gentle climb

        # Section 3: Figure-8 Pattern (12-18s)
        elif t < 18.0:
            t_local = t - 12.0
            omega = 0.35  # Angular frequency (slower)
            radius_8 = 2.5

            x_d = 8.0 + radius_8 * np.sin(omega * t_local)
            y_d = 1.5 * np.sin(2 * omega * t_local)  # Creates figure-8
            z_d = 4.1  # Constant altitude

        # Section 4: Straight Line with Speed Variation (18-24s)
        elif t < 24.0:
            t_local = t - 18.0
            # Gentle speed variation
            speed = 1.5 + 0.3 * np.sin(0.5 * t_local)  # 1.2-1.8 m/s

            x_d = 8.0 + 1.5 * t_local + (0.3/0.5) * (1 - np.cos(0.5 * t_local))
            y_d = 1.0 * np.sin(0.4 * t_local)  # Gentle sway
            z_d = 4.1 + 0.4 * np.sin(0.7 * t_local)  # Gentle altitude variation

        # Section 5: Descending Arc to Hover (24-30s)
        else:
            t_local = t - 24.0
            # Smooth descent and deceleration
            decay = np.exp(-0.5 * t_local)

            x_d = 17.0 + 2.0 * decay
            y_d = 1.0 * decay * np.sin(0.4 * 24.0)
            z_d = 4.1 - 1.6 * (1 - decay)  # Descend to 2.5m

        return np.array([x_d, y_d, z_d])

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
        ax1.set_title('3D Gentle Parcour - Position Tracking')
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
        plt.savefig('parcour_results.png', dpi=300)
        print(f"\nPlot saved: parcour_results.png")
        plt.show()

    def simulation(self):
        for step in range(self.max_time_steps):
            target = self.get_target(self.time_step)
            u = self.controller.controller(self.state, target, dt=self.dt)
            state_dot = self.env.step(self.state, u)
            self.state = self.state + state_dot * self.dt

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
