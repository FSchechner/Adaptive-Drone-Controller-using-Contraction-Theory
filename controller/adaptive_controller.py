import numpy as np

class QuadcopterController:
    def __init__(self,
                 mass=1.2,
                 g=9.81,
                 Kp_long=0.9890, Kd_long=0.8130,
                 Kp_lat=0.5693,  Kd_lat=0.5836,
                 Kp_z=5.0347,    Kd_z=2.9939,
                 Kp_att=17.0805, Kd_att=4.7259,
                 max_tilt_deg=35.0,
                 F_max=25.0,
                 tau_max=3.5,
                 Ixx=0.028, Iyy=0.028, Izz=0.055,
                 enable_adaptation=True ):

        # True (nominal) parameters for baseline control
        self.m_true = mass
        self.g = g
        self.m = mass  # Will use m_hat if adaptation enabled

        # Position gains in body-aligned frame (longitudinal, lateral)
        self.Kp_long = Kp_long
        self.Kd_long = Kd_long
        self.Kp_lat  = Kp_lat
        self.Kd_lat  = Kd_lat
        self.integrate = 0.0

        # Altitude gains
        self.Kp_z = Kp_z
        self.Kd_z = Kd_z

        # Attitude gains (used for phi, theta, psi)
        self.Kp_att = Kp_att
        self.Kd_att = Kd_att

        # Limits
        self.max_tilt = np.radians(max_tilt_deg)
        self.F_max = F_max
        self.tau_max = tau_max

        # === ADAPTIVE CONTROL PARAMETERS ===
        self.enable_adaptation = enable_adaptation

        # Estimated parameters (will be adapted online)
        self.m_hat = mass  # Mass estimate (kg)
        self.D_hat = np.zeros(3)  # Drag coefficient estimates [Dx, Dy, Dz] (kg/s)
        self.I_hat = np.array([Ixx, Iyy, Izz])  # Inertia estimates (kg·m²)

        # Parameter bounds for projection (Eq. 24-26)
        self.m_min = 0.5
        self.m_max = 3.0
        self.d_max = 1.0
        self.I_min = 0.01
        self.I_max = 0.2

        # Adaptation gains (Γ matrix from Eq. 23)
        self.gamma_m = 0.05  # Mass adaptation rate
        self.gamma_D = np.array([0.02, 0.02, 0.02])  # Drag adaptation rates
        self.gamma_I = np.array([0.005, 0.005, 0.005])  # Inertia adaptation rates

        # Store previous values for numerical differentiation
        self.vel_prev = None
        self.omega_prev = None

    def _rotation_matrix(self, phi, theta, psi):
        """Compute rotation matrix from body to world frame (R_WB)"""
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        R = np.array([
            [cpsi*ctheta, cpsi*stheta*sphi - spsi*cphi, cpsi*stheta*cphi + spsi*sphi],
            [spsi*ctheta, spsi*stheta*sphi + cpsi*cphi, spsi*stheta*cphi - cpsi*sphi],
            [-stheta, ctheta*sphi, ctheta*cphi]
        ])
        return R

    def _L_operator(self, a):
        """
        Compute L(a) such that I*a = L(a)*θ_rot
        where θ_rot = [Ixx, Iyy, Izz] (diagonal inertia)
        For diagonal inertia: L(a) = diag(a)
        """
        return np.diag(a)

    def _translational_regressor(self, F_total, phi, theta, psi, vel_dot):
        """
        Compute translational regressor φ_trans from Eq. (21)

        φ_trans = [[R*e3*F]^T, -ξ̇]^T

        Parameters represent: θ_trans = [1/m, Dx/m, Dy/m, Dz/m]^T
        So: acceleration =φ_trans^T * θ_trans
        """
        R = self._rotation_matrix(phi, theta, psi)
        e3 = np.array([0, 0, 1])

        # Thrust direction in world frame
        thrust_world = R @ (e3 * F_total)

        # Regressor: [thrust_world; -vel_dot] (6x1 vector)
        # This gives: Δaccel = φ_trans^T * [1/m, Dx/m, Dy/m, Dz/m]^T
        phi_trans = np.concatenate([thrust_world, -vel_dot])

        return phi_trans  # Shape: (6,)

    def _rotational_regressor(self, omega, omega_dot_ref):
        """
        Compute rotational regressor φ_rot from Eq. (22)

        φ_rot = L(ω̇_r) + [ω]_× L(ω_r)

        where ω_r is reference angular velocity
        Returns 3x3 matrix such that: torque_error = φ_rot * θ_rot
        """
        # Skew-symmetric matrix [ω]_×
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])

        # For hovering, ω_r ≈ 0, so we use actual omega as approximation
        L_omega_dot_ref = self._L_operator(omega_dot_ref)
        L_omega = self._L_operator(omega)

        phi_rot = L_omega_dot_ref + omega_skew @ L_omega

        return phi_rot  # Shape: (3, 3)

    def _project_parameters(self):
        """Project estimated parameters to valid bounds (Eq. 24-26)"""
        # Mass bounds
        self.m_hat = np.clip(self.m_hat, self.m_min, self.m_max)

        # Drag bounds (non-negative)
        self.D_hat = np.clip(self.D_hat, 0.0, self.d_max)

        # Inertia bounds
        self.I_hat = np.clip(self.I_hat, self.I_min, self.I_max)

    def get_parameter_estimates(self):
        """Return current parameter estimates for logging/analysis"""
        return {
            'm_hat': self.m_hat,
            'D_hat': self.D_hat.copy(),
            'I_hat': self.I_hat.copy()
        }

    def reset_adaptation(self):
        """Reset parameter estimates to initial values"""
        self.m_hat = self.m_true
        self.D_hat = np.zeros(3)
        self.I_hat = np.array([0.028, 0.028, 0.055])  # Default values
        self.vel_prev = None
        self.omega_prev = None

    def controller(self, state, target_pos, target_vel=None, dt=0.01):
        x, y, z       = state[0:3]
        vx, vy, vz    = state[3:6]
        phi, theta, psi = state[6:9]
        omega_x, omega_y, omega_z = state[9:12]

        x_d, y_d, z_d = target_pos

        # Default to position-hold mode (zero desired velocity)
        if target_vel is None:
            vx_d, vy_d, vz_d = 0.0, 0.0, 0.0
        else:
            vx_d, vy_d, vz_d = target_vel

        # Horizontal position errors 
        ex = x_d - x
        ey = y_d - y

        # Heading control
        distance = np.sqrt(ex**2 + ey**2)
        eps = 1e-5

        if distance > eps:
            psi_d = np.arctan2(ey, ex)
            e_psi = psi_d - psi
            while e_psi > np.pi:
                e_psi -= 2.0 * np.pi
            while e_psi < -np.pi:
                e_psi += 2.0 * np.pi
        else:
            psi_d = psi
            e_psi = 0.0

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Longitudinal and lateral position errors
        e_long =  cos_psi * ex + sin_psi * ey
        e_lat  = -sin_psi * ex + cos_psi * ey

        # Body-frame velocities (actual)
        v_long =  cos_psi * vx + sin_psi * vy
        v_lat  = -sin_psi * vx + cos_psi * vy

        # Body-frame velocities (desired)
        v_long_d =  cos_psi * vx_d + sin_psi * vy_d
        v_lat_d  = -sin_psi * vx_d + cos_psi * vy_d

        # Velocity errors in body frame
        ev_long = v_long_d - v_long
        ev_lat  = v_lat_d - v_lat

        theta_d = self.Kp_long * e_long + self.Kd_long * ev_long

        # Roll left/right for lateral motion
        phi_d   = -self.Kp_lat * e_lat - self.Kd_lat * ev_lat

        # Tilt limits
        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)
        phi_d   = np.clip(phi_d,   -self.max_tilt, self.max_tilt)

        # --- Altitude control (world z) ---
        e_z = z_d - z
        ev_z = vz_d - vz
        az_d = self.Kp_z * e_z + self.Kd_z * ev_z
        Fz   = self.m * (az_d + self.g)                 

        cos_theta = np.cos(theta)
        cos_phi   = np.cos(phi)
        denom = cos_theta * cos_phi
        denom = np.clip(denom, 0.2, 1.0)

        F_total = Fz / denom
        F_total = np.clip(F_total, 0.0, self.F_max)

        # --- Attitude control
        tau_phi   = self.Kp_att * (phi_d   - phi)   - self.Kd_att * omega_x
        tau_theta = self.Kp_att * (theta_d - theta) - self.Kd_att * omega_y
        tau_psi   = self.Kp_att * (e_psi)           - self.Kd_att * omega_z

        # Saturate torques to prevent numerical instability
        tau_phi   = np.clip(tau_phi,   -self.tau_max, self.tau_max)
        tau_theta = np.clip(tau_theta, -self.tau_max, self.tau_max)
        tau_psi   = np.clip(tau_psi,   -self.tau_max, self.tau_max)

        # === ADAPTATION LAW (Eq. 23) ===
        if self.enable_adaptation:
            vel = np.array([vx, vy, vz])
            omega = np.array([omega_x, omega_y, omega_z])

            # Compute velocity derivatives (numerical differentiation)
            if self.vel_prev is not None:
                vel_dot = (vel - self.vel_prev) / dt
                omega_dot = (omega - self.omega_prev) / dt
            else:
                vel_dot = np.zeros(3)
                omega_dot = np.zeros(3)

            # Velocity errors: e_v = v_desired - v_actual
            if target_vel is not None:
                vx_d, vy_d, vz_d = target_vel
            else:
                vx_d, vy_d, vz_d = 0.0, 0.0, 0.0

            e_v_trans = np.array([vx_d - vx, vy_d - vy, vz_d - vz])

            # Angular velocity error (assume desired ω ≈ 0 for hovering)
            omega_d = np.zeros(3)
            e_v_rot = omega_d - omega

            # === Translational Parameter Adaptation ===
            phi_trans = self._translational_regressor(F_total, phi, theta, psi, vel_dot)

            # Adaptation for θ_trans = [1/m, Dx/m, Dy/m, Dz/m]
            # Simplified: adapt each component based on error magnitude
            error_norm = np.linalg.norm(e_v_trans)

            # Update 1/m (θ₁)
            theta_1_hat = 1.0 / self.m_hat
            theta_1_dot = -self.gamma_m * phi_trans[0] * error_norm
            theta_1_hat += theta_1_dot * dt
            self.m_hat = 1.0 / np.clip(theta_1_hat, 1.0/self.m_max, 1.0/self.m_min)

            # Update D/m (θ₂₋₄)
            theta_D_hat = self.D_hat / self.m_hat
            theta_D_dot = -self.gamma_D * phi_trans[3:6] * error_norm
            theta_D_hat += theta_D_dot * dt
            self.D_hat = theta_D_hat * self.m_hat

            # === Rotational Parameter Adaptation ===
            omega_dot_ref = np.zeros(3)  # Reference angular acceleration (hovering)
            phi_rot = self._rotational_regressor(omega, omega_dot_ref)

            # Adaptation for θ_rot = [Ixx, Iyy, Izz]
            # θ̇_rot = -Γ * φ_rot^T * e_v_rot
            error_rot_norm = np.linalg.norm(e_v_rot)
            theta_I_dot = -self.gamma_I * np.diag(phi_rot) * error_rot_norm

            self.I_hat += theta_I_dot * dt

            # === Parameter Projection ===
            self._project_parameters()

            # Store for next iteration
            self.vel_prev = vel.copy()
            self.omega_prev = omega.copy()

        return np.array([F_total, tau_phi, tau_theta, tau_psi])