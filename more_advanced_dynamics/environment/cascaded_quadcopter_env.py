import numpy as np
from Quadcopter_Dynamics import environment
from Drone import Drone, Drone_with_Package

class CascadedQuadcopter:
    def __init__(self,
                 drone_class=None,
                 constant_disturbance=None,
                 initial_pos=None,
                 initial_vel=None):

        if drone_class is None:
            drone = Drone()
        else:
            drone = drone_class()

        self.env = environment(mass=drone.m, Ixx=drone.Ixx, Iyy=drone.Iyy, Izz=drone.Izz)

        if constant_disturbance is None:
            self.constant_disturbance = np.array([0.0, 0.0, 0.0])
        else:
            self.constant_disturbance = np.array(constant_disturbance)

        self.state = np.zeros(12)

        if initial_pos is not None:
            self.state[0:3] = np.array(initial_pos)

        if initial_vel is not None:
            self.state[3:6] = np.array(initial_vel)

    def step(self, u, dt):
        state_dot = self.env.step(self.state, u)

        state_dot[3:6] += self.constant_disturbance / self.env.m

        self.state = self.state + state_dot * dt

        return self.state

    def get_state(self):
        return self.state.copy()

    def get_pos(self):
        return self.state[0:3].copy()

    def get_vel(self):
        return self.state[3:6].copy()

    def reset(self, pos=None, vel=None):
        self.state = np.zeros(12)

        if pos is not None:
            self.state[0:3] = np.array(pos)

        if vel is not None:
            self.state[3:6] = np.array(vel)
