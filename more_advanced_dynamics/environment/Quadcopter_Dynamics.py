import numpy as np

class environment:
    def __init__(self, mass=1.2, Ixx=0.028, Iyy=0.028, Izz=0.055):
        self.m = mass
        self.g = 9.81
        self.I = np.diag([Ixx, Iyy, Izz])
        self.I_inv = np.diag([1/Ixx, 1/Iyy, 1/Izz])

        self.D = np.diag([0.1, 0.1, 0.15])

        # Rotational damping (aerodynamic drag on rotation)
        self.D_rot = np.diag([0.008, 0.008, 0.01])

        self.e_g = np.array([0, 0, -self.g])

        self.state = np.zeros(12)

        self.lag_history = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])

        self.thrust_noise_std = 0.5
        self.torque_noise_std = 0.01

    def _rotation_matrix(self, phi, theta, psi):
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        R = np.array([
            [cpsi*ctheta, cpsi*stheta*sphi - spsi*cphi, cpsi*stheta*cphi + spsi*sphi],
            [spsi*ctheta, spsi*stheta*sphi + cpsi*cphi, spsi*stheta*cphi - cpsi*sphi],
            [-stheta, ctheta*sphi, ctheta*cphi]
        ])

        return R

    def _euler_rate_matrix(self, phi, theta):
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        # Avoid numerical blow-up but do NOT raise
        eps = 1e-3
        if np.abs(ctheta) < eps:
            ctheta = np.sign(ctheta) * eps if ctheta != 0.0 else eps

        ttheta = stheta / ctheta

        W = np.array([
            [1.0, sphi * ttheta, cphi * ttheta],
            [0.0, cphi,         -sphi],
            [0.0, sphi / ctheta, cphi / ctheta],
        ])
        return W

    def _drift_dynamics(self, state):
        vel = state[3:6]
        euler = state[6:9]
        omega = state[9:12]

        phi, theta = euler[0], euler[1]

        pos_dot = vel

        drag_force = -self.D @ vel
        vel_dot = self.e_g + drag_force / self.m

        W = self._euler_rate_matrix(phi, theta)
        euler_dot = W @ omega

        I_omega = self.I @ omega
        gyro_term = np.cross(omega, I_omega)
        damping_torque = self.D_rot @ omega
        omega_dot = -self.I_inv @ (gyro_term + damping_torque)

        f = np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])

        return f

    def _input_matrix(self, state):
        euler = state[6:9]
        phi, theta, psi = euler

        R = self._rotation_matrix(phi, theta, psi)
        e3 = np.array([0, 0, 1])

        thrust_dir = R @ e3

        B = np.zeros((12, 4))

        B[3:6, 0] = thrust_dir / self.m

        B[9:12, 1:4] = self.I_inv

        return B

    def step(self, state, u):
        # Lag and noise disabled for now
        # u = self.lag(u)
        # u = self.addnoise(u)

        f = self._drift_dynamics(state)

        B = self._input_matrix(state)

        state_dot = f + B @ u

        if state[2] <=0:
            state_dot[2] = max(state_dot[2],0)
            state_dot[5]= max(state_dot[5],0)

        return state_dot
    
    def lag (self, u):
        lag_u = self.lag_history[1,:]
        self.lag_history[1,:] = self.lag_history[0,:]
        self.lag_history[0,:] = u
        return lag_u 
    
    def addnoise(self, u):
        noisy_u = u.copy()

        noisy_u[0] += np.random.normal(0, self.thrust_noise_std)

        noisy_u[1:4] += np.random.normal(0, self.torque_noise_std, 3)

        return noisy_u