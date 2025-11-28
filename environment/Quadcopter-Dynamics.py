import numpy as np

class QuadcopterEnv():
    """
    Quadcopter dynamics environment with uncertain mass and drag.
    State: [position, velocity, euler_angles, body_rates]
    Control: [total_thrust, torque_x, torque_y, torque_z]
    """

    def __init__(self, mass=1.0, drag_coeff=0.1, inertia=None):
        """
        Initialize quadcopter environment.

        Parameters:
        -----------
        mass : float
            Quadcopter mass (kg) - uncertain parameter
        drag_coeff : float
            Linear drag coefficient (N·s/m) - uncertain parameter
        inertia : np.array (3,)
            Moments of inertia [Ixx, Iyy, Izz] (kg·m²)
        """
        # Physical parameters 
        self.m = mass
        self.d = drag_coeff

        # Inertia matrix 
        if inertia is None:
            self.I = np.array([0.01, 0.01, 0.02])  # [Ixx, Iyy, Izz]
        else:
            self.I = np.array(inertia)

        # Constants
        self.g = 9.81  # Gravity (m/s²)

        # State variables
        self.position = np.array([0.0, 0.0, 0.0])      # [x, y, z] in world frame (m)
        self.velocity = np.array([0.0, 0.0, 0.0])      # [vx, vy, vz] in world frame (m/s)
        self.euler = np.array([0.0, 0.0, 0.0])         # [roll, pitch, yaw] (rad)
        self.body_rates = np.array([0.0, 0.0, 0.0])    # [p, q, r] in body frame (rad/s)

        # Control inputs [thrust, tau_x, tau_y, tau_z]
        self.u = np.array([self.m * self.g, 0.0, 0.0, 0.0])

        # Time
        self.t = 0.0

    def step(self, control_input, dt=0.01):
        """
        Advance dynamics by dt using Euler integration.

        Parameters:
        -----------
        control_input : np.array (4,)
            [total_thrust, torque_x, torque_y, torque_z]
        dt : float
            Time step (s)
        """
        self.u = np.array(control_input)

        # Extract control
        thrust = self.u[0]
        torques = self.u[1:4]  # [tau_x, tau_y, tau_z]

        # Get rotation matrix from euler angles (ZYX convention)
        R = self._rotation_matrix(self.euler)

        # ===== Translational Dynamics =====
        # Forces in body frame
        thrust_body = np.array([0, 0, thrust])

        # Transform thrust to world frame
        thrust_world = R @ thrust_body

        # Drag force (linear, in world frame, opposing velocity)
        drag_force = -self.d * self.velocity

        # Gravity force
        gravity_force = np.array([0, 0, -self.m * self.g])

        # Total acceleration in world frame
        acceleration = (thrust_world + drag_force + gravity_force) / self.m

        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # ===== Rotational Dynamics =====
        # Euler's equations: I·ω̇ = τ - ω × (I·ω)
        p, q, r = self.body_rates
        Ixx, Iyy, Izz = self.I

        # Angular accelerations in body frame
        p_dot = (torques[0] - (Izz - Iyy) * q * r) / Ixx
        q_dot = (torques[1] - (Ixx - Izz) * p * r) / Iyy
        r_dot = (torques[2] - (Iyy - Ixx) * p * q) / Izz

        body_rates_dot = np.array([p_dot, q_dot, r_dot])

        # Update body rates
        self.body_rates += body_rates_dot * dt

        # ===== Euler Angle Kinematics =====
        # Convert body rates to euler rate
        euler_dot = self._body_rates_to_euler_rates(self.euler, self.body_rates)

        # Update euler angles
        self.euler += euler_dot * dt

        # Normalize angles to [-pi, pi]
        self.euler = np.arctan2(np.sin(self.euler), np.cos(self.euler))

        # Update time
        self.t += dt

        return self.get_state()

    def get_state(self):
        """Return current state as dictionary."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'euler': self.euler.copy(),
            'body_rates': self.body_rates.copy(),
            't': self.t
        }

    def reset(self, position=None, velocity=None, euler=None, body_rates=None):
        """Reset environment to initial state."""
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else np.array(position)
        self.velocity = np.array([0.0, 0.0, 0.0]) if velocity is None else np.array(velocity)
        self.euler = np.array([0.0, 0.0, 0.0]) if euler is None else np.array(euler)
        self.body_rates = np.array([0.0, 0.0, 0.0]) if body_rates is None else np.array(body_rates)
        self.u = np.array([self.m * self.g, 0.0, 0.0, 0.0])
        self.t = 0.0

        return self.get_state()

    def _rotation_matrix(self, euler):
        """
        Compute rotation matrix from Euler angles (ZYX convention).

        Parameters:
        -----------
        euler : np.array (3,)
            [roll, pitch, yaw] angles in radians

        Returns:
        --------
        R : np.array (3, 3)
            Rotation matrix from body to world frame
        """
        phi, theta, psi = euler  # roll, pitch, yaw

        # Rotation matrices
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0,            0,            1]
        ])

        R_y = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        R_x = np.array([
            [1, 0,           0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)]
        ])

        # Combined rotation (ZYX order)
        R = R_z @ R_y @ R_x

        return R

    def _body_rates_to_euler_rates(self, euler, body_rates):
        """
        Convert body frame angular rates to Euler angle rates.

        Parameters:
        -----------
        euler : np.array (3,)
            [roll, pitch, yaw] angles
        body_rates : np.array (3,)
            [p, q, r] body angular rates

        Returns:
        --------
        euler_dot : np.array (3,)
            [roll_dot, pitch_dot, yaw_dot]
        """
        phi, theta, _ = euler
        p, q, r = body_rates

        # Transformation matrix
        W = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi),               -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])

        euler_dot = W @ np.array([p, q, r])

        return euler_dot

    def set_parameters(self, mass=None, drag_coeff=None, inertia=None):
        """Update uncertain parameters (for testing adaptation)."""
        if mass is not None:
            self.m = mass
        if drag_coeff is not None:
            self.d = drag_coeff
        if inertia is not None:
            self.I = np.array(inertia)



