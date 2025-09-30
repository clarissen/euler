import numpy as np
import numba 
from numba import njit, jit

#--------------------
# generic objects
#--------------------
# =========================================================================
# converts rank 2 tensor indices (a,b) to a row-flattened vector index (c)
@njit
def flat_index(dim, row, col):
    return dim*row + col

# flat spacetime metric, row-flattened
gmn = np.array([[-1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]).flatten()
g_mn = gmn # raised = lowered


#--------------------
# OUTPUT functions
#--------------------
# =========================================================================

def get_flux(q):

    # would it be faster to write the functions explicitly here?
    v = get_v(q)
    ep = get_ep(q,v)
    P = get_P(ep)

    f = np.zeros(q.shape)
    f[0] = q[1]
    f[1] = get_Tmn_2x2(q,v,P)[flat_index(2,1,1)]

    return f

def get_src(q):
    return np.zeros(q.shape)

#--------------------
# Euler Hydro functions
#--------------------
# =========================================================================

# energy-momentum tensor from state q, only need the t,x elements here so its 2x2
def get_Tmn_2x2(q,v,P):
    return np.array([ q[0], q[1], \
                      q[1], q[1] * v + P] )

# to initialize the energy-momentum tensor from ep,v ICs
def make_Tmn_2x2(v, ep, P, gamma):
    return np.array([ (ep + P) * gamma ** 2 - P, (ep + P) * gamma**2 * v, \
                     (ep + P) * gamma**2 * v, (ep + P) * gamma**2 * v**2 + P])

# x-velocity
@njit
def get_v(q):
    # return 3 * q[1] / (2 * q[0] + np.sqrt(4 * q[0]**2 - 3*q[1]**2))
    # v0 = M / ( E (w+1)) = 3 * q[1] / (q[0]*4)
    # w = P/ep = 1/3 for ideal hydro EOS
    # M = q[1]
    # E = q[0]
    # \frac{2v_0}{1 + \sqrt{1 - 4 w v_0^2} }
    # return 2 * (3 * q[1] / (q[0]*4)) / (1 + np.sqrt(1 - 4.0/3.0 * (3 * q[1] / (q[0]*4))**2))
    # \frac{2v_0}{1 + \sqrt{1 - 4 w v_0^2} }
    v0 = 3 * q[1] / (q[0]*4)
    return 2 * v0 / (1 + np.sqrt(1 - 4.0/3.0 * v0**2))

# energy density
@njit
def get_ep(q,v):
    return q[0] - q[1] * v

# Equation of state
@njit
def get_P(ep):
    return ep / 3

# gamma factor
def gamma(v):
    return 1/np.sqrt(1-v**2)

# four velocity
def get_u(v,gamma):
    return np.array([gamma, gamma * v, np.zeros(len(v)), np.zeros(len(v))])



# -----------------------------------------------------