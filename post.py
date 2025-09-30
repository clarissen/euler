import numpy as np
import json

# homemade modules
import euler as hydro
import data_manager as data_m
import kurganovtadmor as kt

def calculate_vars(sim_name):
    # importing simulation data
    param_dict, post_dict, x, t_arr, qtx = data_m.load_sim(sim_name, False)

    print("performing post-simulation calculations to obtain more physical variables...")

    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # post-processing calculations

    T00_tx = qtx[:,0]
    T0x_tx = qtx[:,1]

    eptx = []
    vtx = []
    Ptx = []
    gammatx = []

    Txx_tx = []
    depdx_tx = []
    dvdx_tx = []


    for i in range(0,len(t_arr)):
        vtx.append(hydro.get_v(qtx[i]))
        gammatx.append(hydro.gamma(vtx[i]))

    vtx = np.array(vtx)

    for i in range(0,len(t_arr)):
        eptx.append(hydro.get_ep(qtx[i], vtx[i]))   

    eptx = np.array(eptx)

    Ptx = eptx/3

    for i in range(0, len(t_arr)):
        Txx_tx.append(hydro.get_Tmn_2x2(qtx[i], vtx[i], Ptx[i])[hydro.flat_index(2,1,1)] )
        depdx_tx.append(kt.general_ddx(eptx[i], theta_kt, hx))
        dvdx_tx.append(kt.general_ddx(vtx[i], theta_kt, hx))  

    Txx_tx = np.array(Txx_tx)
    depdx_tx = np.array(depdx_tx)
    dvdx_tx = np.array(dvdx_tx)

    vars = ["T00(t,x)", T00_tx, "T0x(t,x)", T0x_tx, "Txx(t,x)", Txx_tx, "ep(t,x)", eptx, \
            "depdx(t,x)", depdx_tx, "v(t,x)", vtx, "dvdx(t,x)", dvdx_tx, "P(t,x)", Ptx, \
                "gamma(t,x)", gammatx]

    # variables = ["v(t,x)", vtx, "ep(t,x)", eptx, "P(t,x)", Ptx]
    print("...calculations complete.")
    data_m.save_npy_list(sim_name, vars, "vars")

    return vars

# calculate_vars(data_m.load_check("sims"))