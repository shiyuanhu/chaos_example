# A driven damped pendulum system that can evolve many 
# initial conditions by broadcasting
#
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 

import numpy as np
from RK4 import RK4
from block_entropy import block_entropy
import matplotlib
import matplotlib.pyplot as plt

class pendulum_entropy:

    def __init__(self, y0, dt=1e-2, omega_D=2./3, q=2, g=1.125):
        """
            y0: initial conditons of shape 2 by npts. y[0] is the 
                angular velocity and y[1] is angle
            npts: number of initial conditions
            dt: time step
            omega_D: angular frequency of external driven torque
            g: amplitude of external torque
            q: mass
        """
        _,self.npts = y0.shape
        self.y = np.copy(y0)
        # parameters
        self.dt = dt
        self.omega_D = omega_D
        self.q = q
        self.g = g 
        
    def deri(self,y,t):
        dy = np.copy(y)
        dy[0] = -y[0]/self.q-np.sin(y[1])+self.g*np.cos(self.omega_D*t)
        dy[1] = y[0]
        return dy

    def return_map(self):
        """
            Return the angle back to [-pi,pi]
        """
        idx1 = self.y[1]>np.pi
        idx2 = self.y[1]<np.pi
        
        if np.sum(idx1) > 0:
            self.y[1][idx1] = self.y[1][idx1]-np.floor((self.y[1][idx1]+np.pi)/(2*np.pi))*2*np.pi
        if np.sum(idx2) > 0:
            self.y[1][idx2] = self.y[1][idx2]-np.ceil((self.y[1][idx2]-np.pi)/(2*np.pi))*2*np.pi

    def update(self, tfinal):
        """
            Evolve the dynamic equation and store the data
        """
        t = 0; kk = 0
        nstep = int(np.round(tfinal/self.dt))+1 # number of time steps
        self.omega = np.zeros((nstep,self.npts))
        self.theta = np.zeros((nstep,self.npts))

        while t <(tfinal+1e-10):
            self.return_map()
            self.omega[kk] = self.y[0]
            self.theta[kk] = self.y[1]

            self.y = RK4(self.y, self.dt, t, self.deri)
            kk += 1; t += self.dt

        return self

if __name__ == "__main__": 
    eps = 1e-2 # size of the unit cell
    npts = 400 # number of points
    
    y0 = np.zeros((2,npts))
    # Randomly distribute initial conditions inside one unit cell
    # In order to get a smooth function of entropy versus time, 
    # average over different unit cells is necessary

    # Here, computation is done only for one unit cell located at (0.1,0.1)
    y0[0,:] = 0.1+np.random.rand(npts)*eps
    y0[1,:] = 0.1+np.random.rand(npts)*eps
    
    b = pendulum_entropy(y0, omega_D=2./3, q=4, g=1.5)
    b.update(50)
    nstep = len(b.omega)
    entropy = np.zeros(nstep)

    for i in range(nstep):
        data = np.vstack((b.omega[i],b.theta[i]))
        entropy[i] = block_entropy(data, epsilon=eps)
        
    print(entropy)