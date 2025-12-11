import numpy as np 


class Drone: 
    def __init__(self):
        self.Ixx = 0.028
        self.Iyy = 0.028 
        self.Izz = 0.045
        self.m = 1.9 
        self.F_max = 60
        self.tau_max = 4

class Drone_with_Package: 
    def __init__(self):
        self.Ixx = 0.0542 
        self.Iyy = 0.0542 
        self.Izz = 0.045 
        self.m = 2.9 
        self.F_max = 60
        self.tau_max = 4
        self.d = np.array([0.0,0.0,0.0])
