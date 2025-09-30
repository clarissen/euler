import sys
# commonly changed parameters

# problem type
problem_list = ["gaussian", "ep-shock", "v-shock", "riemann-test"]
problem = problem_list[1]

# mesh
# ==========================
x0 = True # include x = 0 in grid?
cCFL = 0.5 # courant factor <=0.5

hx = 0.0125/4 # step sizes (0.2,0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4,0.0125/8)
ht = 0.5*hx # for now make this a fixed (SMALL) number independent of hx so that time steps line up in convergence test
if ht > cCFL*hx:
    print("CFL Condition not met. Exiting program.")
    sys.exit()

N = 75  # length of grid
iterations = N/(ht) 
tmod = int(1/ht)
# ==========================

# viscosity
Neta = 0

# conformal
ep_coeff = 10 # alpha in ep = alpha * T^4

# Kurganov-Tadmor
theta_kt = 1.0

# metadata 
# ==========================
# alarms on or off?
alarms = False
# automatically pass checks and prints and generate file names for cluster?
sucomp = True
# animation bounds?
anim_bounds = True
# zoom in on animations?
# anim_zoom = False
# ==========================