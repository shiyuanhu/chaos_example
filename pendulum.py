# A driven damped pendulum system
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 
# You may use, share, or modify this file freely

import numpy as np
from RK4 import RK4
from gsr import gsr

class pend_lyapunov:
    """
        Compute the lyapunov spectra of driven damped pendulum
        The same notation is used as "G. L. Baker and J. P. Gollub. 
        Chaotic dynamics: an introduction. Cambridge university 
        press, 1996.". Some values of Lyapunov exponents are listed 
        in Table 5.1 in Chapter 5.
    """
    def __init__(self, y0, dt=1e-2, omega_D=2./3, q=2, g=1.125):
        """
            y0: initial conditons of shape 2 by npts
            npts: number of initial conditions
            dt: time step
            omega_D: angular frequency of external driven torque
            g: amplitude of external torque
            q: mass
        """
        self.y = np.zeros(2)
        self.y[0] = y0[0] # angular velocity / omega
        self.y[1] = y0[1] # angle / theta
        
        # parameters
        self.dt = dt
        self.omega_D = omega_D
        self.q = q
        self.g = g 

    def deri(self,y,t):
        """
            Derivate of the phase space variables dz/dt
        """
        dy = np.zeros(2)
        dy[0] = -y[0]/self.q-np.sin(y[1])+self.g*np.cos(self.omega_D*t)
        dy[1] = y[0]
        return dy
    
    def jacobi(self,y):
        """
            Compute the jacobian matrix J = d(deri(z))/dz
        """
        j11 = -1/self.q
        j12 = -np.cos(y[1])
        j21 = 1; j22 = 0
        J = np.array([[j11,j12],[j21,j22]])
        return J

    def update(self, tfinal, tau=1):
        """
            tau should be around one period
        """
        def deriv_pert(A, t):
            """
                Time derivative of the perturbation
            """
            J = self.jacobi(self.y)
            return J.dot(A)

        r = int(np.round(tfinal/tau))
        nstep = int(np.round(tfinal/self.dt))+1
        
        k = 0; kk = 0; t = 0.0
        A = np.identity(2)
        self.norms = np.zeros((r,2))
        ifirst = 1

        while t < (tfinal+1e-8):
            self.y = RK4(self.y, self.dt, t, self.deri)
            A = RK4(A, self.dt, t, deriv_pert)

            t_mul = t/tau
            if (np.isclose(t_mul-int(round(t_mul)), 0)) and (ifirst == 0):
                A, norm = gsr(A)
                self.norms[kk,:] = norm
                kk += 1

            t += self.dt; k += 1
            ifirst = 0

        self.lya = np.sum(np.log(self.norms), axis=0)/(r*tau)
        return self

b = pend_lyapunov(np.array([0,1]), omega_D=2./3, q=4, g=1.5)
b.update(1000, tau=1);
print(b.lya)
