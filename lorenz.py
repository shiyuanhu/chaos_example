# Lorenz system
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 
import numpy as np
from RK4 import RK4
from gsr import gsr

class lorenz:
    """
        Compute the lyapunov spectra of the Lorentz system
        We follow the notations in "Physica D 139, 72–86 (2000)"
    """
    def __init__(self, z0, dt=1e-2, sigma=10, rho=28, beta=8./3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.z = np.zeros(3)
        self.z[0] = z0[0]; self.z[1] = z0[1]; self.z[2] = z0[2]

    def deri(self,z,t):
        dz = np.zeros(3)
        dz[0] = self.sigma*(z[1]-z[0])
        dz[1] = z[0]*(self.rho-z[2])-z[1]
        dz[2] = z[0]*z[1]-self.beta*z[2]

        return dz
    
    def jacobi(self,z):
        """
            Jacobian matrix given by d(deri(z))/dz
        """
        J = np.zeros((3,3))
        J[0,0] = -self.sigma; J[0,1] = self.sigma; J[0,2] = 0
        J[1,0] = self.rho-z[2]; J[1,1] = -1; J[1,2] = -z[0]
        J[2,0] = z[1]; J[2,1] = z[0]; J[2,2] = -self.beta

        return J
    
    def update(self, tfinal, tau=1):
        """
            tau should be around 1 period
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
        A = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.norms = np.zeros((r,3))
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

l = lorenz([0.,1.,0.])
l.update(1000, tau=1)
print(l.lya)