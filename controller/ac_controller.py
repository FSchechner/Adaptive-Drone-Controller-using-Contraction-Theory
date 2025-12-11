import numpy as np

## I used Claude Code for comments and debugging 
class AdaptiveController:
    def __init__(self,
                 m_nominal=1.9,
                 g=9.81,
                 # Defaults set to tuned values (Powell search)
                 lambda_xy=3.6197,
                 lambda_z=10.0,
                 k_xy=14.9999,
                 k_z=15.0,
                 gamma_alpha=0.3038,
                 gamma_d=0.1,
                 alpha_min=0.05,
                 alpha_max=1.0,
                 d_max=20.0,
                 F_max=60.0,
                 dt=0.01):

        self.m_nominal = m_nominal  # nominal mass used to seed α̂
        self.g = g  # gravity constant in the model
        self.dt = dt  # integration step (passed through the sim)

        self.Lambda = np.diag([lambda_xy, lambda_xy, lambda_z])  # Λ in Eq. (14)/(16)
        self.K = np.diag([k_xy, k_xy, k_z])  # K in Eq. (16)

        self.alpha_hat = 1.0 / m_nominal  # initial α̂ = 1/m_nominal
        self.d_hat = np.array([0.0, 0.0, 0.0])  # initial d̂
        self.theta_hat = np.array([self.alpha_hat, 0.0, 0.0, 0.0])  # θ̂ = [α̂, d̂]

        self.Gamma = np.diag([gamma_alpha, gamma_d, gamma_d, gamma_d])  # Γ in Eq. (adaptive_law)

        self.alpha_min = alpha_min  # projection bound on α̂ (lower)
        self.alpha_max = alpha_max  # projection bound on α̂ (upper)
        self.d_max = d_max  # projection bounds on d̂ components

        self.F_max = F_max  # actuator saturation limit

        self.pos_prev = None  # previous p_d for finite differences
        self.vel_prev_d = None  # previous \dot{p}_d for finite differences

    def compute_control(self, pos, vel, pos_d, vel_d=None, acc_d=None, dt=None):
        dt = self.dt if dt is None else dt  # time step used for discrete integration

        # Finite-difference \dot{p}_d when not provided (paper: velocity reference definition)
        if vel_d is None:
            if self.pos_prev is None:
                vel_d = np.array([0.0, 0.0, 0.0])
            else:
                vel_d = (pos_d - self.pos_prev) / dt

        # Finite-difference \ddot{p}_d when not provided (paper: acceleration reference)
        if acc_d is None:
            if self.vel_prev_d is None:
                acc_d = np.array([0.0, 0.0, 0.0])
            else:
                acc_d = (vel_d - self.vel_prev_d) / dt

        e_pos = pos - pos_d  # e_p in Eq. (14)
        e_vel = vel - vel_d  # e_v in Eq. (14)

        s = e_vel + self.Lambda @ e_pos  # sliding surface s (Eq. (14))

        # Commanded acceleration a_cmd = -Λ e_v - K s + \ddot{p}_d (Eq. (16))
        a_cmd = -self.Lambda @ e_vel - self.K @ s + acc_d

        e3 = np.array([0.0, 0.0, 1.0])  # e3 unit vector
        a_des = a_cmd + self.g * e3  # a_des = a_cmd + g e3 (Eq. (17))

        m_hat = 1.0 / max(self.alpha_hat, 1e-6)  # \hat{m} = 1/\hat{\alpha}

        F_control = m_hat * (a_des - self.d_hat)  # F = \hat{m}(a_des - d̂) (Eq. adaptive_force)
        F_control = np.clip(F_control, -self.F_max, self.F_max)  # force saturation

        # Regressor Y(F) as in Eq. (Y)
        Y = np.zeros((3, 4))
        Y[:, 0] = F_control
        Y[:, 1:4] = np.eye(3)

        theta_dot = self.Gamma @ (Y.T @ s)  # θ̇ = Γ Yᵀ s (Eq. adaptive_law)

        self.theta_hat += dt * theta_dot  # integrate θ̇ with dt

        self.theta_hat[0] = np.clip(self.theta_hat[0], self.alpha_min, self.alpha_max)  # proj α̂
        self.theta_hat[1] = np.clip(self.theta_hat[1], -self.d_max, self.d_max)  # proj d̂_x
        self.theta_hat[2] = np.clip(self.theta_hat[2], -self.d_max, self.d_max)  # proj d̂_y
        self.theta_hat[3] = np.clip(self.theta_hat[3], -self.d_max, self.d_max)  # proj d̂_z

        self.alpha_hat = self.theta_hat[0]  # cache α̂
        self.d_hat = self.theta_hat[1:4]  # cache d̂

        self.pos_prev = pos_d  # store p_d[k] for next diff
        self.vel_prev_d = vel_d  # store \dot{p}_d[k] for next diff

        return F_control  # applied control force

    def reset(self):
        self.alpha_hat = 1.0 / self.m_nominal
        self.d_hat = np.array([0.0, 0.0, 0.0])
        self.theta_hat = np.array([self.alpha_hat, 0.0, 0.0, 0.0])
        self.pos_prev = None
        self.vel_prev_d = None

    def get_estimates(self):
        m_hat = 1.0 / max(self.alpha_hat, 1e-6)
        return {
            'm_hat': m_hat,
            'alpha_hat': self.alpha_hat,
            'd_hat': self.d_hat.copy(),
            'theta_hat': self.theta_hat.copy()
        }
