import numpy as np

class AttitudeController:
    def __init__(self,
                 Kp_phi=29.44,
                 Kd_phi=5.18,
                 Kp_theta=35.43,
                 Kd_theta=1.00,
                 Kp_psi=15.10,
                 Kd_psi=1.87,
                 psi_d=0.0):

        self.Kp_att = np.array([Kp_phi, Kp_theta, Kp_psi])
        self.Kd_att = np.array([Kd_phi, Kd_theta, Kd_psi])
        self.psi_d = psi_d
        self.F_max = 40.0
        self.tau_max = 4.0

    def compute_control(self, F_des, state):
        euler = state[6:9]
        omega = state[9:12]

        phi, theta, psi = euler

        F_des_norm = np.linalg.norm(F_des)

        # Compute desired angles from F_des (unchanged)
        if F_des_norm > 1e-6:
            phi_d = np.arcsin(-F_des[1] / F_des_norm)
            theta_d = np.arctan2(F_des[0], F_des[2])
        else:
            phi_d = 0.0
            theta_d = 0.0

        # Compute body z-axis in inertial frame using CURRENT attitude
        # R = Rz(psi) * Ry(theta) * Rx(phi)
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # Third column of rotation matrix (body z in inertial frame)
        z_B = np.array([
            cpsi * sth * cphi + spsi * sphi,
            spsi * sth * cphi - cpsi * sphi,
            cth * cphi
        ])

        # Compute thrust to maintain desired vertical force at current attitude
        # F_thrust * z_B[2] = F_des[2]
        if abs(z_B[2]) > 1e-3:  # Avoid division by zero
            F_thrust = F_des[2] / z_B[2]
        else:
            # Fallback to norm if nearly inverted
            F_thrust = F_des_norm

        F_thrust = np.clip(F_thrust, 0.0, self.F_max)

        phi_d = np.clip(phi_d, -np.pi/4, np.pi/4)
        theta_d = np.clip(theta_d, -np.pi/4, np.pi/4)

        euler_d = np.array([phi_d, theta_d, self.psi_d])
        omega_d = np.array([0.0, 0.0, 0.0])

        e_att = euler - euler_d
        e_omega = omega - omega_d

        e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))

        tau = -self.Kp_att * e_att - self.Kd_att * e_omega
        tau = np.clip(tau, -self.tau_max, self.tau_max)


        u = np.array([F_thrust, tau[0], tau[1], tau[2]])

        return u