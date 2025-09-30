import numpy as np

import config

# SIMULATION CONFIGURATION
# =========================================================================

# Cartesian Mesh creation (t,x)
#--------------------
hx = config.hx
N = config.N # number of steps

c = config.cCFL # CFL condition ht <= c * hx
ht = config.ht

# initial time
t0 = 0.0
# grid specs in x
xpos = np.arange(hx,N+hx,hx) # xpos = [+hx,...,50]
xneg = -xpos[::-1] # reverse "::-1" and negative "-"

# the whole grid
x = np.concatenate((xneg, [0], xpos)) # number of cells always odd and centered at 0

# 
xL = x[0]
xR = x[-1]
x0 = 0
xM = x[ int((len(x)-1)/2) ]
Ncells = len(x)

#--------------------


# Kurganov-Tadmor params
#--------------------
# indexing, need two ghost points for KT numerical gradients
Km2 = np.arange(0, Ncells - 4)
Km1 = np.arange(1, Ncells - 3)
K   = np.arange(2, Ncells - 2)
Kp1 = np.arange(3, Ncells - 1)
Kp2 = np.arange(4, Ncells)

#python list for access
indexing = [Km2, Km1, K, Kp1, Kp2]

theta_kt = config.theta_kt # kt numerical dissipation limiter
    # 1.0 most numerical dissipation, smallest spatial gradients in solution
    # 2.0 least numerical disipation, largest spatial gradients in solution

max_local_prop_speed = 0.0 # max local prop speed !=1 and will be calculated in kurganovtadmor.py

#--------------------


# CHANGE-ABLES
#--------------------
#what problem do i want to run?
problem_list = config.problem_list
problem = config.problem

# scheme -- there's only one right now
scheme = "kt-tvdr2k"
# how many time steps do i want to take?
iterations = config.iterations
# alarms on or off?
alarms = config.alarms
# sucomp on or off?
sucomp = config.sucomp
#--------------------

# Useful information
#--------------------
units = "GeV"
invunits = "1/GeV"
hbarc = 0.1973
cs = 1.0 / np.sqrt(3.0) # conformal speed of sound
spacetime = "cartesian"
#--------------------

# problem specifics 
#--------------------
if problem == problem_list[0]:
    epmax = 0.15
    epR = epmax
    epmin = 0.1
    epL = epmin

    vL = 0
    vR = vL

    Delta = 0

if problem == problem_list[1]:
    # distribution in epsilon only with L and R states
    epL = 1.3
    epR = 0.3
    vL = 0
    vR = vL

    Delta = 1.0 # scale of L,R state separation


if problem == problem_list[2]:
    # distribution in velocity only with L and R states
    epR = 0.1
    epL = epR
    vL = 0.99
    vR = 0.8

    Delta = 1.0 # scale of L,R state separation

if problem == problem_list[3]:
    # choose shock IC in v
    vL = 0.3
    vR = 0.15
    # choose the value in front of wave for ep
    epR = 0.1

    # constrain IC in ep behind shock front according to IC in v and epR
    # Mach et al - PRE.81.046313 Eq (39)
    def gamma(v):
        return 1/np.sqrt(1-v**2)
    W = gamma(vL)
    W_bar = gamma(vR)
    kap = cs**2 /(1 + cs**2)
    Theta = W**2 * W_bar**2 * (vL - vR)**2 / (2 * kap * (1 - kap))

    epL = epR * (1 + Theta + np.sqrt( (1+Theta)**2.0 - 1) )

    Delta = 0.0
#--------------------

# =========================================================================

# Dictionary which contains global variables that will never change and may used throughout the sim
param_dict = {"cs": cs, "a": max_local_prop_speed, "theta_kt": theta_kt, "hx": hx, "c": c, "ht": ht}

#Dictionary which contains global variables that will never change, is unique to the problem, 
# and will only be used in POST-PROCESSING or INITIALIZATION
post_dict = {"hbarc": hbarc, "cs": cs, "epL": epL, "epR": epR, "vL": vL, "vR": vR, "Delta": Delta, "Ncells": Ncells, \
             "xL": xL, "xR": xR, "xM": xM, "t0": t0, "iterations": iterations, "spacetime": spacetime, "problem": problem}

specs = "spacetime = " + str(spacetime) + ", problem = " + str(problem) + "\n" \
        + "(epL, epR) = (" + str(epL) + ", " + str(epR) + ")" + \
        ", (vL, vR) = (" + str(vL) + ", " + str(vR) + ")" \
        + ", shock width Delta = " + str(Delta) + "\n" \
        + "theta_kt = " + str(theta_kt) + ", hx = " + str(hx) + ", ht = " + str(ht) \
         + ', c (CFL) = ' + str(c)  + ', Ncells = ' + str(Ncells) + ", grid = [" + str(xL) + "," + str(xR) + "]" + "\n" \
        + "max local propagation speed = " + str(max_local_prop_speed) + ", alarms = " + str(alarms)

version = problem +"_"+ scheme

sim_config = [f"Simulation Configuration:", \
              f"This code simulates the {problem} problem in {spacetime} spacetime using a {scheme} scheme.", \
              f"Spatial resolution of hx = {hx} {invunits} on a grid = [{xL}, {xR}] {invunits}.", \
                f"Temporal resolution of ht = {ht} {invunits}.", \
                f" ------------------------- ", \
                 f"All parameters: {param_dict}", \
                    f"{post_dict}"]

# predetermiend unique sim name
unique = "euler_" + str(problem) \
        + "_(epL,epR)=(" + str(epL) + ","+ str(epR) + ")"\
        + "_(vL,vR)=(" + str(vL) + "," + str(vR) + ")" \
        + "_Delta=" + str(Delta) \
        + "_hx=" + str(hx) + "_ht=" + str(ht) \
        + '_Ncells=' + str(Ncells) + "_grid=[" + str(xL) + "," + str(xR) + "]" \
        + "_c+=" + str(max_local_prop_speed) \
        + "_theta_kt=" + str(theta_kt) + "_alarms=" + str(alarms) + "_sucomp=" + str(sucomp)

# print(unique)