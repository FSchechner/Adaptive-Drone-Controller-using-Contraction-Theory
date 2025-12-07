import numpy as np
import sys
sys.path.insert(0, '../Drone')
from Drone_1 import Drone

class ContractionAdaptiveQuadController:
    """Contraction-based adaptive controller. Adapts mass (α=1/m) and 3D disturbance (d_a)."""

    def __init__(self,
                 g=9.81,
                 Kp_xy=1.5, Kd_xy=1.0,
                 Kp_z=5.0,  Kd_z=3.0,
                 Kp_att=15.0, Kd_att=4.0,
                 max_tilt_deg=35.0,
                 lambda_xy=2.0, lambda_z=2.0,
                 k_xy=3.0, k_z=3.0,
                 gamma_alpha=2.3711, gamma_dx=0.0100, gamma_dy=0.0100, gamma_dz=1.0008,
                 sigma=0.0,
                 m_min=0.5, m_max=3.0, d_max=10.0):

        self.Drone = Drone()
        self.m_nominal = self.Drone.m
        self.m_true = self.Drone.m
        self.g = g
        self.F_max = self.Drone.F_max
        self.tau_max = self.Drone.tau_max
        self.max_tilt = np.radians(max_tilt_deg)

        self.Kp_att = Kp_att
        self.Kd_att = Kd_att
        self.I = np.array([self.Drone.Ixx, self.Drone.Iyy, self.Drone.Izz])

        self.Kp_xy = Kp_xy
        self.Kd_xy = Kd_xy
        self.Kp_z = Kp_z
        self.Kd_z = Kd_z

        # Λ = diag(λ_xy, λ_xy, λ_z), K = diag(k_xy, k_xy, k_z)
        self.Lambda = np.diag([lambda_xy, lambda_xy, lambda_z])
        self.K = np.diag([k_xy, k_xy, k_z])

        # Γ = diag(γ_α, γ_dx, γ_dy, γ_dz)
        self.Gamma = np.diag([gamma_alpha, gamma_dx, gamma_dy, gamma_dz])

        # σ-modification for robustness (leak term)
        self.sigma = sigma

        # α ∈ [1/m_max, 1/m_min]
        self.alpha_min = 1.0 / m_max
        self.alpha_max = 1.0 / m_min
        self.d_max = d_max

        # θ̂ = [α̂, d̂_ax, d̂_ay, d̂_az]ᵀ
        self.theta_hat = np.array([1.0 / self.m_nominal, 0.0, 0.0, 0.0])

        # Store nominal for σ-modification
        if sigma > 0:
            self.theta_nominal = self.theta_hat.copy()
        else:
            self.theta_nominal = None

        self.estimate_history = {
            'time': [], 'alpha_hat': [], 'm_hat': [],
            'd_hat_x': [], 'd_hat_y': [], 'd_hat_z': []
        }
        self.psi_d = None
        self.target_pos_prev = None

    def reset(self):
        """Reset adaptive parameters to nominal values."""
        self.theta_hat = np.array([1.0 / self.m_nominal, 0.0, 0.0, 0.0])
        if self.sigma > 0:
            self.theta_nominal = self.theta_hat.copy()
        self.psi_d = None
        self.target_pos_prev = None
        self.estimate_history = {
            'time': [], 'alpha_hat': [], 'm_hat': [],
            'd_hat_x': [], 'd_hat_y': [], 'd_hat_z': []
        }

    def controller(self, state, target_pos, dt):
        pos = state[0:3]
        vel = state[3:6]
        phi, theta, psi = state[6:9]
        omega = state[9:12]

        if self.psi_d is None:
            self.psi_d = psi

        if self.target_pos_prev is not None:
            v_d = (target_pos - self.target_pos_prev) / dt
        else:
            v_d = np.zeros(3)

        self.target_pos_prev = target_pos.copy()

        e = pos - target_pos
        e_v = vel - v_d

        s = e_v + self.Lambda @ e
        a_cmd = - self.Lambda @ e_v - self.K @ s

        alpha_hat = self.theta_hat[0]
        d_hat = self.theta_hat[1:4]

        R = self._rotation_matrix(phi, theta, psi)
        e3 = np.array([0.0, 0.0, 1.0])
        b = R @ e3

        a_tmp = a_cmd - self.g * e3 - d_hat
        F_total = np.dot(b, a_tmp) / max(alpha_hat, 1e-6)  # Prevent division by zero
        F_total = np.clip(F_total, 0.0, self.F_max)

        # Desired attitude
        thrust_vec_des = a_cmd - self.g * e3 - d_hat
        thrust_norm = np.linalg.norm(thrust_vec_des)
        z_b_des = thrust_vec_des / thrust_norm if thrust_norm > 0.1 else np.array([0.0, 0.0, 1.0])

        phi_d, theta_d = self._attitude_from_thrust_direction(z_b_des, self.psi_d)
        phi_d = np.clip(phi_d, -self.max_tilt, self.max_tilt)
        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)

        # PD attitude control
        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = self._wrap_angle(self.psi_d - psi)
        omega_d = np.zeros(3)

        tau_phi = self.Kp_att * e_phi + self.Kd_att * (omega_d[0] - omega[0])
        tau_theta = self.Kp_att * e_theta + self.Kd_att * (omega_d[1] - omega[1])
        tau_psi = self.Kp_att * e_psi + self.Kd_att * (omega_d[2] - omega[2])

        tau_phi = np.clip(tau_phi, -self.tau_max, self.tau_max)
        tau_theta = np.clip(tau_theta, -self.tau_max, self.tau_max)
        tau_psi = np.clip(tau_psi, -self.tau_max, self.tau_max)

        # === ADAPTIVE LAW ===
        # NOTE: Mass (α) and vertical disturbance (d_az) are NOT uniquely identifiable during hover.
        # - At hover: thrust ≈ mg, so α̂·F ≈ g + d̂_az creates coupling
        # - Rich motion (varying acceleration) improves convergence
        # - Recommendation: Use slow adaptation gains (current optimized values are tuned)

        Re3F = R @ e3 * F_total
        phi = np.hstack([Re3F.reshape(-1, 1), np.eye(3)])

        # Regressor normalization prevents large updates when ||phi|| is large
        phi_norm2 = np.sum(phi ** 2)
        denom = 1.0 + phi_norm2

        # Correct adaptive law with negative gradient
        theta_dot = -self.Gamma @ (phi.T @ s) / denom

        # Optional σ-modification (leak term for robustness)
        if self.sigma > 0:
            leak = self.theta_hat - self.theta_nominal
            theta_dot -= self.sigma * (self.Gamma @ leak)
        self.theta_hat += dt * theta_dot
        self._project_parameters()

        return np.array([F_total, tau_phi, tau_theta, tau_psi])

    def _rotation_matrix(self, phi, theta, psi):
        """R = Rz(ψ) Ry(θ) Rx(φ)"""
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        return np.array([
            [c_psi*c_theta, c_psi*s_theta*s_phi - s_psi*c_phi, c_psi*s_theta*c_phi + s_psi*s_phi],
            [s_psi*c_theta, s_psi*s_theta*s_phi + c_psi*c_phi, s_psi*s_theta*c_phi - c_psi*s_phi],
            [-s_theta, c_theta*s_phi, c_theta*c_phi]
        ])

    def _attitude_from_thrust_direction(self, z_b_des, psi_d):
        x_c = np.array([np.cos(psi_d), np.sin(psi_d), 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_norm = np.linalg.norm(y_b_des)

        if y_norm > 1e-6:
            y_b_des = y_b_des / y_norm
        else:
            y_b_des = np.array([-np.sin(psi_d), np.cos(psi_d), 0.0])

        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack([x_b_des, y_b_des, z_b_des])

        sin_theta_d = np.clip(-R_d[2, 0], -1.0, 1.0)
        theta_d = np.arcsin(sin_theta_d)
        cos_theta_d = np.cos(theta_d)

        phi_d = np.arctan2(R_d[2, 1], R_d[2, 2]) if np.abs(cos_theta_d) > 1e-6 else 0.0
        return phi_d, theta_d

    def _wrap_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def _project_parameters(self):
        self.theta_hat[0] = np.clip(self.theta_hat[0], self.alpha_min, self.alpha_max)
        self.theta_hat[1] = np.clip(self.theta_hat[1], -self.d_max, self.d_max)
        self.theta_hat[2] = np.clip(self.theta_hat[2], -self.d_max, self.d_max)
        self.theta_hat[3] = np.clip(self.theta_hat[3], -self.d_max, self.d_max)

    def record_estimates(self, t):
        self.estimate_history['time'].append(t)
        self.estimate_history['alpha_hat'].append(self.theta_hat[0])
        self.estimate_history['m_hat'].append(1.0 / self.theta_hat[0])
        self.estimate_history['d_hat_x'].append(self.theta_hat[1])
        self.estimate_history['d_hat_y'].append(self.theta_hat[2])
        self.estimate_history['d_hat_z'].append(self.theta_hat[3])

    def get_estimates(self):
        return {
            'alpha_hat': self.theta_hat[0],
            'm_hat': 1.0 / self.theta_hat[0],
            'm_true': self.m_true,
            'm_error_pct': abs(1.0/self.theta_hat[0] - self.m_true) / self.m_true * 100,
            'd_hat': self.theta_hat[1:4].copy(),
            'theta_hat': self.theta_hat.copy()
        }