import numpy as np
class PDController:
    def __init__(self,
                 m_nominal=1.9,
                 g=9.81,
                 Kp_xy=30.0,  # Optimized (Powell) for simplified dynamics
                 Kd_xy=8.6398,
                 Kp_z=30.0,
                 Ki_xy=0.0,
                 Ki_z=0.0699,
                 Kd_z=8.5386,
                 dt=0.01):
    
        self.m_nominal = m_nominal  # nominal mass m_nom in Eq. (pid_force)
        self.g = g  # gravity constant g
        self.Kp = np.array([Kp_xy,Kp_xy, Kp_z])  # K_p gains in Eq. (pd_accel)
        self.Kd = np.array([Kd_xy,Kd_xy, Kd_z])  # K_d gains in Eq. (pd_accel)
        self.Ki = np.array([Ki_xy,Ki_xy, Ki_z])  # integral gains (not in paper, for bias)
        self.integrate = np.array([0.0, 0.0, 0.0])  # integral state
        self.max_integrate = 10.0  # clamp integral windup
        self.dt = dt  # integration step (passed from sim)
        self.pos_prev = None  # previous p_d for finite difference \dot{p}_d
        self.F_max = 60  # actuator saturation (consistent with paper limit)

    def compute_control(self, pos, vel, pos_d, vel_d=None, acc_d=None, dt=None):
        dt = self.dt if dt is None else dt  # time step used for discrete updates

        # Finite-difference \dot{p}_d when not provided (paper velocity reference)
        if vel_d is None:
            if self.pos_prev is None:
                vel_d = np.array([0.0, 0.0, 0.0])
            else:
                vel_d = (pos_d - self.pos_prev) / dt
        
        e_pos = pos - pos_d  # e_p in Eq. (pd_accel)
        e_vel = vel - vel_d  # e_v in Eq. (pd_accel)
        self.integrate += e_pos * dt  # integral of position error (bias rejection)
        self.integrate = np.clip(self.integrate,-self.max_integrate, self.max_integrate)  # anti-windup

        a_des = -self.Kp * e_pos - self.Kd * e_vel - self.Ki * self.integrate  # a_des = -Kp e_p - Kd e_v (Eq. pd_accel)

        e3 = np.array([0.0, 0.0, 1.0])  # e3 unit vector
        a_total = a_des + self.g * e3  # a_total = a_des + g e3 (gravity compensation)

        F_control = self.m_nominal * a_total  # F = m_nom (a_total) (Eq. pid_force)
        self.pos_prev = pos_d  # store p_d[k] for next diff
        F_control = np.clip(F_control, -self.F_max, self.F_max)  # actuator saturation
        return F_control
