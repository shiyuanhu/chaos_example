import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(42)

def gsr(A):
    """
        Gram–Schmidt orthogonalization of the columns of input matrix A
    """
    def proj(v,u):
        """
            compute the projection of v onto u
        """
        prefc = v.dot(u)/u.dot(u)
        return prefc*u

    _,ncol = A.shape
    vnorms = np.zeros(ncol) 
    vnorms[0] = np.linalg.norm(A[:,0]) # normalized the first column

    for i in range(1,ncol): # iteration starts at the second column
        Ai = A[:,i]
        Ai_new = np.copy(Ai)
        for j in range(0, i): # subtract the projection of Ai on the previous orthonormal columns 
            Aj = A[:,j]
            Ai_new = Ai_new-proj(Ai,Aj)
        vnorms[i] = np.linalg.norm(Ai_new)
        A[:,i] = Ai_new
    return A/vnorms, vnorms

def RK4(y, dt, t, deri):
    """
        4th-order Runge-Kutta scheme
    """
    k1 = dt*deri(y,t)
    k2 = dt*deri(y+0.5*k1,t)
    k3 = dt*deri(y+0.5*k2,t)
    k4 = dt*deri(y+k3,t)
    y += (k1+2.0*k2+2.0*k3+k4)/6.0
    return y

class pend_lyapunov:
    """
        Compute the lyapunov spectra of driven damped pendulum
        We use the same notations as "G. L. Baker and J. P. Gollub. 
        Chaotic dynamics: an introduction. Cambridge university 
        press, 1996.". Some values of Lyapunov exponents are listed 
        in Table 5.1 in Chapter 5.
    """
    def __init__(self, y0, dt, omega_D=2./3, q=2, g=1.125):
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

    def update(self, tfinal, frac=0.1):
        def deriv_pert(A, t):
            """
                Time derivative of the perturbation
            """
            J = self.jacobi(self.y)
            return J.dot(A)

        tau = frac*tfinal; r = int(np.round(1/frac))
        nstep = int(np.round(tfinal/self.dt))+1
        
        k = 0; kk = 0; t = 0.0
        A = np.identity(2)
        self.norms = np.zeros((r,2))
        ifirst = 1

        while t < (tfinal+1e-8):
            self.y = euler(self.y, self.dt, t, self.deri)
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

b = pend_lyapunov(np.array([0,1]), 1e-2, omega_D=2./3, q=4, g=1.5)
b.update(1000, frac=1e-3);
b.lya

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
        self.z = np.zeros(2)
        self.z[0] = z0[0]; self.z[1] = z0[1]

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
    
    def update(self, tfinal, frac=0.1):

        def deriv_pert(A, t):
            """
                Time derivative of the perturbation
            """
            J = self.jacobi(self.z)
            return J.dot(A)
        
        tau = frac*tfinal; r = int(np.round(1/frac))
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
v.update(2e3, frac=1e-4)
v.lya

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
    
    def update(self, tfinal, frac=0.1):

        def deriv_pert(A, t):
            """
                Time derivative of the perturbation
            """
            J = self.jacobi(self.z)
            return J.dot(A)
        
        tau = frac*tfinal; r = int(np.round(1/frac))
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
l.update(2000, frac=0.001)
l.lya
