import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

import data_manager as data_m

#--------------------
# Functions 
#--------------------
# =========================================================================

def get_L1_errors(phi_coarse, phi_medium, phi_fine, hx_list, ht_list):
    """ Computes discrete L1 norm between two fields defined on the same grid. """

    hx_coarse, hx_medium, hx_fine = hx_list
    ht_coarse, ht_medium, ht_fine = ht_list

    rx_cm = int(round(hx_coarse/hx_medium))
    rx_cf = int(round(hx_coarse/hx_fine))

    delta_cm = (phi_coarse - phi_medium[:,::rx_cm])
    delta_mf = (phi_medium[:,::rx_cm] - phi_fine[:,::rx_cf])

    # <-- fix here: ignore NaNs
    L1_cm = np.nansum(np.abs(delta_cm), axis=1) * hx_coarse 
    L1_mf = np.nansum(np.abs(delta_mf), axis=1) * hx_coarse
    return L1_cm, L1_mf


def get_L2_errors(phi_coarse, phi_medium, phi_fine, hx_list, ht_list):
    """ Computes discrete L2 norm between two fields defined on the same grid. """

    hx_coarse, hx_medium, hx_fine = hx_list
    ht_coarse, ht_medium, ht_fine = ht_list

    rx_cm = int(round(hx_coarse/hx_medium))
    rx_cf = int(round(hx_coarse/hx_fine))

    delta_cm = (phi_coarse - phi_medium[:,::rx_cm])
    delta_mf = (phi_medium[:,::rx_cm] - phi_fine[:,::rx_cf])

    a, b = 0, -1

    # <-- fix here: ignore NaNs
    L2_cm = np.sqrt(np.nansum(delta_cm[:,a:b]**2, axis=1) * hx_coarse)
    L2_mf = np.sqrt(np.nansum(delta_mf[:,a:b]**2, axis=1) * hx_coarse)
    return L2_cm, L2_mf


def get_convergence_Q(L2_coarse_medium, L2_medium_fine):
    tol = 0
    return L2_coarse_medium/( L2_medium_fine + tol)

def get_convergence_Q_log2(L2_coarse_medium, L2_medium_fine):
    tol = 0
    return np.log2(L2_coarse_medium/(L2_medium_fine + tol) )

def plot_convergence(sim_name, problem, t_array, hx_list, ht_list, Q_list, var_name_list, var_label_list, Q_type):
    
    plt.rcParams.update({
    # "axes.titlesize": 16,     # subplot titles
    "axes.labelsize": 14,     # x and y labels
    # "xtick.labelsize": 12,    # tick labels
    # "ytick.labelsize": 12,
    "legend.fontsize": 10
    })

    sim_loc = "./sims/" + sim_name
    gct_loc = sim_loc + "/gct/"
    
    if not os.path.exists(gct_loc ):
        os.makedirs(gct_loc)

    hx_c, hx_m, hx_f = hx_list
    ht_c, ht_m, ht_f = ht_list

    linestyles = ['k', 'b', 'm', 'r', 'g']
    
    plt.figure() # not plotting early times

    for i in range(0, len(Q_list)):
        plt.plot(t_array[2:], Q_list[i][2:], linestyles[i], label = var_label_list[i])
        if Q_type == 'L1_log2':
            np.save(gct_loc+ Q_type + "_" + var_name_list[i], Q_list[i])
    
    plt.xlabel(r"$t$ (1/GeV)")
    plt.ylabel(r"Q(t)")

    plt.ylim(0, 2.1)

    plt.legend(loc = 'lower left')

    # plt.savefig(gct_loc + Q_type + "_" +problem+ "_frame=" + str(frame) +"_hx="+str(hx_c)+","+str(hx_m)+","+str(hx_f)+"_etas="+str(etaovers)+"_"+var_name_list[0]+"_"+var_name_list[1]+".pdf")
    plt.savefig(gct_loc + Q_type +".pdf")

    print("saved under: " + gct_loc)

def build_mask(arr, width=2):
    """
    Build a boolean mask for a [t, x] array, masking around the 
    max in x for each t.

    Parameters
    ----------
    arr : np.ndarray
        Shape (T, X). Used to locate maxima along x for each t.
    width : int
        Number of points on each side of the maximum to mask.

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape (T, X). 
        True = keep, False = mask out.
    max_positions : np.ndarray
        Array of shape (T,) with the x-index of the max for each time.
    """
    T, X = arr.shape
    max_positions = np.argmax(arr, axis=1)  # (T,)
    mask = np.ones((T, X), dtype=bool)

    for t in range(T):
        i = max_positions[t]
        left = max(0, i - width)
        right = min(X, i + width + 1)
        mask[t, left:right] = False

    return mask, max_positions



# =========================================================================


#--------------------
# Dataframe
#--------------------
# =========================================================================
data = {
    # CONVENTION 
    #                       list
    # name of list : [coarse sim, medium sim, fine sim]

    # epshock

    "ep-shock, euler, ep=(1.3,0.3), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=1.0": ["euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                         "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                         "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True"
                                                                                        ],
    

    "ep-shock, euler, ep=(1.3,0.3), hx = 0.05,0.025,0.0125, ht = 0.5hx, minmod3, Delta=1.0": ["euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                              "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.025_ht=0.0125_Ncells=6001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                              "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.0125_ht=0.00625_Ncells=12001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True"
                                                                                               ],

    "ep-shock, euler, ep=(1.3,0.3), hx = 0.025,0.0125,0.0125/2, ht = 0.5hx, minmod3, Delta=1.0": ["euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.025_ht=0.0125_Ncells=6001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                              "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.0125_ht=0.00625_Ncells=12001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                              "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.00625_ht=0.003125_Ncells=24001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True"
                                                                                               

    ],                                                                                      

    "ep-shock, euler, ep=(1.3,0.3), hx = 0.0125,0.0125/2,0.0125/4, ht = 0.5hx, minmod3, Delta=1.0": ["euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.0125_ht=0.00625_Ncells=12001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                                     "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.00625_ht=0.003125_Ncells=24001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                                     "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.003125_ht=0.0015625_Ncells=48001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True"
                                                                                                     ],

    # theta_KT = 2.0
    "ep-shock, euler, thetaKT = 2, ep=(1.3,0.3), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=1.0": ["euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.0_theta_kt=2.0_alarms=False_sucomp=True",
                                                                                         "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.0_theta_kt=2.0_alarms=False_sucomp=True",
                                                                                         "euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.0_theta_kt=2.0_alarms=False_sucomp=True"
                                                                                        ],

}                
# =========================================================================

#--------------------
# Loading 
#--------------------
# =========================================================================


df = pd.DataFrame(data)
runs = ['gaussian', 'ep-shock', 'v-gaussian']

run = input("ep shock or gaussian or v shock or gaussian w momentum? type g, e, vg, v: ")

if run == 'g':
    df_type = 'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.4,d=64*0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    sim_c = df.loc[0, df_type]
    sim_m = df.loc[1, df_type]
    sim_f = df.loc[2, df_type]
    problem = 'gaussian'

if run == 'e':
    df_type = "ep-shock, euler, ep=(1.3,0.3), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=1.0"
    # df_type = "ep-shock, euler, thetaKT = 2, ep=(1.3,0.3), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=1.0"
    # df_type = "ep-shock, euler, ep=(1.3,0.3), hx = 0.05,0.025,0.0125, ht = 0.5hx, minmod3, Delta=1.0"
    # df_type = "ep-shock, euler, ep=(1.3,0.3), hx = 0.025,0.0125,0.0125/2, ht = 0.5hx, minmod3, Delta=1.0"
    # df_type = "ep-shock, euler, ep=(1.3,0.3), hx = 0.0125,0.0125/2,0.0125/4, ht = 0.5hx, minmod3, Delta=1.0"
    sim_c = df.loc[0, df_type]
    sim_m = df.loc[1, df_type]
    sim_f = df.loc[2, df_type]
    problem = 'ep-shock'


# loading data
# ===============================
sim_names = [sim_c, sim_m, sim_f]

# Global Variables may be used in convergence test
# coarse data
param_dict_c, post_dict_c, x_c, t_c, qtx_c = data_m.load_sim(sim_names[0], False)
vars_c, var_names_c = data_m.load_vars2(sim_names[0], False) 
# medium data
param_dict_m, post_dict_m, x_m, t_m, qtx_m = data_m.load_sim(sim_names[1], False)
vars_m, var_names_m = data_m.load_vars2(sim_names[1], False)
# fine data
param_dict_f, post_dict_f, x_f, t_f, qtx_f = data_m.load_sim(sim_names[2], False)
vars_f, var_names_f = data_m.load_vars2(sim_names[2], False)

t_list = [t_c, t_m, t_f]
print("time")
print(t_list[0])
print(t_list[1])
print(t_list[2])

x_list = [x_c, x_m, x_f]
print("space")
print(x_list[0])
print(x_list[1])
print(x_list[2])

# resolutions
hx_c = param_dict_c["hx"]
hx_m = param_dict_m["hx"]
hx_f = param_dict_f["hx"]
hx_list = [hx_c, hx_m, hx_f]

ht_c = param_dict_c["ht"]
ht_m = param_dict_m["ht"]
ht_f = param_dict_f["ht"]
ht_list = [ht_c, ht_m, ht_f]

cCFL = param_dict_c["c"]
# frame = param_dict_c["frame"]
# etaovers = round(param_dict_c["etaovers"],3)

epL = post_dict_c["epL"]
epR = post_dict_c["epR"]

# ===============================

# General plotting parameters
var_name_list = ["T00(t,x)","T0x(t,x)", "Txx(t,x)", "ep(t,x)", "v(t,x)"]
var_label_list = [r"$T^{00}(t,x)$", r"$T^{0x}(t,x)$", r"$T^{xx}(t,x)$", r"$\varepsilon(t,x)$", r"$v(t,x)$"]
t_array = t_c

# T00, T0x, ep, v
# -------------------
# coarse
T00_tx_c = qtx_c[:,0]
T0x_tx_c = qtx_c[:,1]
ep_tx_c = vars_c[7]
v_tx_c = vars_c[11]
Txx_tx_c = vars_c[5]

depdx_tx_c = vars_c[9]


# medium
T00_tx_m = qtx_m[:,0]
T0x_tx_m = qtx_m[:,1]
ep_tx_m = vars_m[7]
v_tx_m = vars_m[11]
Txx_tx_m = vars_m[5]

depdx_tx_m = vars_m[9]

# fine
T00_tx_f = qtx_f[:,0]
T0x_tx_f = qtx_f[:,1]
ep_tx_f = vars_f[7]
v_tx_f = vars_f[11]
Txx_tx_f = vars_f[5]

depdx_tx_f = vars_f[9]

# coarse
mask_c = build_mask(depdx_tx_c, width=3)

T00_tx_c_masked = np.where(mask_c, T00_tx_c, np.nan)
T0x_tx_c_masked = np.where(mask_c, T0x_tx_c, np.nan)
ep_tx_c_masked  = np.where(mask_c, ep_tx_c, np.nan)
v_tx_c_masked   = np.where(mask_c, v_tx_c, np.nan)
Txx_tx_c_masked = np.where(mask_c, Txx_tx_c, np.nan)

# medium
mask_m = build_mask(depdx_tx_m, width=3)

T00_tx_m_masked = np.where(mask_m, T00_tx_m, np.nan)
T0x_tx_m_masked = np.where(mask_m, T0x_tx_m, np.nan)
ep_tx_m_masked  = np.where(mask_m, ep_tx_m, np.nan)
v_tx_m_masked   = np.where(mask_m, v_tx_m, np.nan)
Txx_tx_m_masked = np.where(mask_m, Txx_tx_m, np.nan)

# fine
mask_f = build_mask(depdx_tx_f, width=3)

T00_tx_f_masked = np.where(mask_f, T00_tx_f, np.nan)
T0x_tx_f_masked = np.where(mask_f, T0x_tx_f, np.nan)
ep_tx_f_masked  = np.where(mask_f, ep_tx_f, np.nan)
v_tx_f_masked   = np.where(mask_f, v_tx_f, np.nan)
Txx_tx_f_masked = np.where(mask_f, Txx_tx_f, np.nan)


# -------------------


# L1 stuff
# ---------
# T00
T00_L1_cm, T00_L1_mf = get_L1_errors(T00_tx_c_masked, T00_tx_m_masked, T00_tx_f_masked, hx_list, ht_list)
Q_T00_log2_L1 = get_convergence_Q_log2(T00_L1_cm, T00_L1_mf)
Q_T00_L1 = get_convergence_Q(T00_L1_cm, T00_L1_mf)

# T0x 
T0x_L1_cm, T0x_L1_mf = get_L1_errors(T0x_tx_c_masked, T0x_tx_m_masked, T0x_tx_f_masked, hx_list, ht_list)
Q_T0x_log2_L1 = get_convergence_Q_log2(T0x_L1_cm, T0x_L1_mf)
Q_T0x_L1 = get_convergence_Q(T0x_L1_cm, T0x_L1_mf)

# ep
ep_L1_cm, ep_L1_mf = get_L1_errors(ep_tx_c_masked, ep_tx_m_masked, ep_tx_f_masked, hx_list, ht_list)
Q_ep_log2_L1 = get_convergence_Q_log2(ep_L1_cm, ep_L1_mf)
Q_ep_L1 = get_convergence_Q(ep_L1_cm, ep_L1_mf)

# v
v_L1_cm, v_L1_mf = get_L1_errors(v_tx_c_masked, v_tx_m_masked, v_tx_f_masked, hx_list, ht_list)
Q_v_log2_L1 = get_convergence_Q_log2(v_L1_cm, v_L1_mf)
Q_v_L1 = get_convergence_Q(v_L1_cm, v_L1_mf)

# Txx
Txx_L1_cm, Txx_L1_mf = get_L1_errors(Txx_tx_c_masked, Txx_tx_m_masked, Txx_tx_f_masked, hx_list, ht_list)
Q_Txx_log2_L1 = get_convergence_Q_log2(Txx_L1_cm, Txx_L1_mf)
Q_Txx_L1 = get_convergence_Q(Txx_L1_cm, Txx_L1_mf)

# all 
Q_list_log2_L1 = [Q_T00_log2_L1, Q_T0x_log2_L1, Q_Txx_log2_L1, Q_ep_log2_L1, Q_v_log2_L1]
Q_list_L1 = [Q_T00_L1, Q_T0x_L1, Q_Txx_L1, Q_v_L1]

# plotting
plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_L1, var_name_list, var_label_list, "L1")
plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_log2_L1, var_name_list, var_label_list, "L1_log2")
# ---------



# # L2 stuff
# # ---------
# # T00
# T00_L2_cm, T00_L2_mf = get_L2_errors(T00_tx_c, T00_tx_m, T00_tx_f, hx_list, ht_list)
# Q_T00_log2_L2 = get_convergence_Q_log2(T00_L2_cm, T00_L2_mf)
# Q_T00_L2 = get_convergence_Q(T00_L2_cm, T00_L2_mf)

# # T0x 
# T0x_L2_cm, T0x_L2_mf = get_L2_errors(T0x_tx_c, T0x_tx_m, T0x_tx_f, hx_list, ht_list)
# Q_T0x_log2_L2 = get_convergence_Q_log2(T0x_L2_cm, T0x_L2_mf)
# Q_T0x_L2 = get_convergence_Q(T0x_L2_cm, T0x_L2_mf)

# # ep
# ep_L2_cm, ep_L2_mf = get_L2_errors(ep_tx_c, ep_tx_m, ep_tx_f, hx_list, ht_list)
# Q_ep_log2_L2 = get_convergence_Q_log2(ep_L2_cm, ep_L2_mf)
# Q_ep_L2 = get_convergence_Q(ep_L2_cm, ep_L2_mf)

# # v
# v_L2_cm, v_L2_mf = get_L2_errors(v_tx_c, v_tx_m, v_tx_f, hx_list, ht_list)
# Q_v_log2_L2 = get_convergence_Q_log2(v_L2_cm, v_L2_mf)
# Q_v_L2 = get_convergence_Q(v_L2_cm, v_L2_mf)

# # Txx
# Txx_L2_cm, Txx_L2_mf = get_L2_errors(Txx_tx_c, Txx_tx_m, Txx_tx_f, hx_list, ht_list)
# Q_Txx_log2_L2 = get_convergence_Q_log2(Txx_L2_cm, Txx_L2_mf)
# Q_Txx_L2 = get_convergence_Q(Txx_L2_cm, Txx_L2_mf)

# # all 
# Q_list_log2_L2 = [Q_T00_log2_L2, Q_T0x_log2_L2, Q_Txx_log2_L2, Q_ep_log2_L2, Q_v_log2_L2]
# Q_list_L2 = [Q_T00_L2, Q_T0x_L2, Q_Txx_L2, Q_v_L2]

# # plotting
# plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_L2, var_name_list, var_label_list, "L2")
# plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_log2_L2, var_name_list, var_label_list, "L2_log2")

# # ---------

# =========================================================================