# Van der pol oscillator system
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 

import numpy as np
from RK4 import RK4
from gsr import gsr

class van_oscillator:

    def __init__(self, z0, dt=1e-2, d=-5,b=5,omega=2.47):
        """
            Compute the lyapunov spectra of van der pol oscillator
            We follow the notations in 'Physica D 139, 72–86 (2000)'
        """
        self.d = d
        self.b = b
        self.omega = omega
        self.dt = dt
        self.z = np.copy(z0)

    def deri(self,z,t):
        """
            Time derivative of the variables
        """
        dz = np.zeros(2)
        dz[0] = z[1]
        dz[1] = -self.d*(1-z[0]**2)*z[1]-z[0]+self.b*np.cos(self.omega*t)
        return dz
    
    def jacobi(self,z):
        """
            Jacobian matrix given by d(deri(z))/dz
        """
        J = np.zeros((2,2))
        J[0,0] = 0; J[0,1] = 1
        J[1,0] = 2*self.d*z[0]*z[1]-1
        J[1,1] = -self.d*(1-z[0]**2)

        return J
    
    def update(self, tfinal, tau=0.1):
        """
            tau should be around one period
        """
        def deriv_pert(A, t):
            """
                Time derivative of the perturbation
            """
            J = self.jacobi(self.z)
            return J.dot(A)
        
        r = int(np.round(tfinal/tau))
        nstep = int(np.round(tfinal/self.dt))+1
        
        k = 0; kk = 0; t = 0.0
        A = np.array([[1.,0.],[0.,1.]])
        self.norms = np.zeros((r,2))
        ifirst = 1

        while t < (tfinal+1e-8):
            self.z = RK4(self.z, self.dt, t, self.deri)
            A = RK4(A, self.dt, t, deriv_pert)

            t_mul = t/tau
            if (np.isclose(t_mul-int(round(t_mul)), 0)) and (ifirst == 0):
                A, norm = gsr(A)
                self.norms[kk,:] = norm
                kk += 1

            t += self.dt; k += 1
            ifirst = 0

        self.lya = np.sum(np.log(self.norms), axis=0)/(r*tau)

        return

v = van_oscillator([-1., 1.])
v.update(2e3, tau=0.2)
v.lya
