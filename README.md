# chaos_example
1. Compute the Lyapunov spectrum of several example sysmtes using the algorithm described in "Meccanica 15.1 (1980): 9-20." by G. Benettin et.al.
This algorithm requires the explicit form of the dynamical equations and the Jacobian matrix. A dynamical system is chaotic if one or more Lyapunov 
exponents are larger than zero.

2. Compute the Shannon entropy of the phase space evolution for driven damped pendulum. Typically, after an initial transient, the entropy increases 
linearly before saturation. The slope of the linear regime is the so-called Kolmogorov-Sinai entropy rate, which equals the sum of the positive 
Lyapunov exponents. The computation needs to solve the dynamical equations for many initial conditions. This method is desired when the Jacobian matrix 
is difficult to compute, for example, when we have integral equations.
