# A 4th-order Runge-Kutta scheme
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 

def RK4(y, dt, t, deri):
    """
        A 4th-order Runge-Kutta scheme
        Input y at time t, compute y at time t+dt 
        deri: dy/dt, which may be a function of time t
    """
    k1 = dt*deri(y,t)
    k2 = dt*deri(y+0.5*k1,t)
    k3 = dt*deri(y+0.5*k2,t)
    k4 = dt*deri(y+k3,t)
    y += (k1+2.0*k2+2.0*k3+k4)/6.0
    return y