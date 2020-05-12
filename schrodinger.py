'''
Numerical solution of the Schrodinger Equation.
'''

import numpy as np
import functools
from scipy.linalg import eigh
import janima

def init_wave(x):
    '''Probability distribution at t=0. Here: harmonic wave enveloped by
    a Gaussian.'''
    x0 = -2
    dx = 0.5
    k0 = 10
    return 1/(dx*np.sqrt(2*np.pi))*np.exp(-0.5*((x-x0)/dx)**2)*np.exp(1j*k0*x)

def sprod(phi, psi, dx):
    '''Standard dot product of two functions (arrays).'''
    return np.sum(dx*np.conjugate(phi)*psi)

def norm(phi, dx):
    '''Standard norm of functions (arrays).'''
    return np.sqrt(dx * sum(abs(phi)**2))
    

def V(x):
    '''Potential.'''
    V0 = 60
    if x>0:
        return V0
    else:
        return 0

def phi_t(x, t, cs, ew, ef, heff):
    '''Time propagated wave'''
    c_t = cs * np.exp(-1j*ew*t/heff)
    phi = np.matmul(c_t, np.transpose(ef))
    return phi
    
if __name__ == "__main__":
    
    heff = 1
    xs, delta_x = np.linspace(-30, 30, 2000, retstep=True)
    ts = np.linspace(0, 1.2, 400)
    N = len(xs)
    
    # Create potential array
    V_vect = np.vectorize(V)
    Vs = V_vect(xs)
    
    z = heff**2/(2.0*delta_x**2)
    
    # Hamilton Matrix
    H = (np.diag(Vs + 2.0*z) +
         np.diag(-z*np.ones(N-1), k=-1) +                  
         np.diag(-z*np.ones(N-1), k=1))
    
    print("Diagonalizing Hamilton matrix ...")
    
    # Diagonalization into eigenvalues and eigenfunctions
    ew, ef = eigh(H)
    
    # Normalize eigenfunctions
    for i in range(len(ew)):
        ef[:, i] /= np.sqrt(delta_x)
    
    # Coefficients array
    cs = ew*(0+0j)
    
    # Set initial condition
    phi0 = init_wave(xs)
    
    # Normalize initial condition
    phi0 /= norm(phi0, delta_x)
    
    # Find coefficients of eigenfunctions for stationary solution
    for i, E in enumerate(ew):
        cs[i]=sprod(ef[:, i], phi0, delta_x)
    
    # Calculate energy
    E = sum(abs(cs)**2*ew)
    
    # Time Propagation
    Psi = functools.partial(phi_t, cs=cs, ew=ew, ef=ef, heff=heff)
    Psi2 = lambda x, t: abs(Psi(x, t))**2+E/max(Vs)
    Vt = lambda x, t: Vs/max(Vs)
    
    # Animation
    
    Animation1 = janima.anima(ts, xs)
    
    Animation1.set_functions([{'func':Psi2, 'subplt':111},
                              {'func':Vt, 'subplt':111}])
    
    Animation1.set_parameters({'axlimits':{111:[[-5, 5], None]},
        'save':True,
        'suptitles': {111:'Reflection and tunneling at a step potential'}})
    
    print("Rendering animation (this may take a while) ...")
    
    Animation1.run()
    