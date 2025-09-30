import numpy as np

# homemade modules
import parameters as params
import euler as hydro

#--------------------
# Functions for initial data
#--------------------
# =========================================================================

def fermidirac(x, x0, fL, fR, Delta):
    return fR + (fL-fR)/(1 + np.exp( (x-x0)/Delta ) )

def gaussian(x,x0,fmax,fmin,width):
    return fmax * np.exp(-(x-x0)**2/width) + fmin

def shock(x,x0,fL,fR):
    u = np.zeros(x.shape)
    
    for j in range(len(x)):

        if x[j] <= x0:
            u[j] = fL
            
        if x[j] > x0:
            u[j] = fR
            
    return u

# =========================================================================

#--------------------
# initialize fluid state functions
#--------------------
# =========================================================================

def init_ep_shock(post_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]

    x0 = post_dict["xM"]

    ep = fermidirac(x,x0,epL,epR,Delta)
    v = vL * np.ones(len(x))

    T00 = hydro.make_Tmn_2x2(v,ep,ep/3,hydro.gamma(v))[0]
    T01 = hydro.make_Tmn_2x2(v,ep,ep/3,hydro.gamma(v))[1]

    return np.array([T00,T01])

def init_v_shock(post_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]

    x0 = post_dict["xM"]

    ep = epL * np.ones(len(x))
    v = fermidirac(x,x0,vL,vR,Delta)

    T00 = hydro.make_Tmn_2x2(v,ep,cs**2*ep,hydro.gamma(v))[0]
    T01 = hydro.make_Tmn_2x2(v,ep,cs**2*ep,hydro.gamma(v))[1]

    return np.array([T00,T01])

def init_gaussian(post_dict,x):
    epmin = post_dict["epL"]
    epmax = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]

    x0 = post_dict["xM"]

    width = 25 # 1/GeV
    ep = gaussian(x,x0,epmax,epmin,width)
    v = vL*np.ones(len(x))

    T00 = hydro.make_Tmn_2x2(v,ep,cs**2*ep,hydro.gamma(v))[0]
    T01 = hydro.make_Tmn_2x2(v,ep,cs**2*ep,hydro.gamma(v))[1]

    return np.array([T00,T01])

def init_riemann(post_dict, x):

    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]

    x0 = post_dict["xM"]

    ep = shock(x,x0,epL,epR)
    v = shock(x,x0,vL,vR)

    T00 = hydro.make_Tmn_2x2(v,ep,cs**2*ep,hydro.gamma(v))[0]
    T01 = hydro.make_Tmn_2x2(v,ep,cs**2*ep,hydro.gamma(v))[1]

    return np.array([T00,T01])



#--------------------
# initialize output
#--------------------
def initial_conditions(problem):

    if problem == params.problem_list[0]:
        return init_gaussian(params.post_dict, params.x)

    if problem == params.problem_list[1]:
        return init_ep_shock(params.post_dict, params.x)
    
    if problem == params.problem_list[2]:
        return init_v_shock(params.post_dict, params.x)
    
    if problem == params.problem_list[3]:
        return init_riemann(params.post_dict, params.x)
# =========================================================================