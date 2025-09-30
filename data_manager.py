import numpy as np
import os 
import json 
import sys
import parameters as params

# CHECKS AND SAVES
# -------------------------------------
def sim_check(specs, sucomp):
    print("you are about to run a sim with specifications...")
    print("-------------")
    print(specs)
    print("-------------")
    if sucomp == True:
        return
    else:
        inp = input("is this correct? type y to continue or n to stop: ")
        if inp == "y":
            pass
        else: 
            sys.exit("code stopped.") 
    

def save_check(dir, unique, sucomp):
    print("saving data in /"+ str(dir) +"/ directory. ")
    if sucomp == True:
        return unique
    else:
        autonaming = input("auto-generate name? y or n:")
        if autonaming == "y":
            return unique
        else:
            return input('please name the /' + str(dir) + "/ folder: ")

    # inp = input("save data in " + str(dir) + " directory? ")
    # if inp == "n":
    #     pass
    # else:
    #     name = input('please name the ' + str(dir) + "folder: ")
    #     return name

# saves all necessary simulation objects
def save_sim(sim_name, jsonfiles, npyfiles, txtfiles):
    path = "./sims/"
    new_path = path + sim_name
    # creates dir if dir not made
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    save_loc = new_path + "/"

    # using json, save dicts
    for i in range(0,len(jsonfiles),2):
        with open(save_loc + jsonfiles[i], 'w') as f:
                json.dump(jsonfiles[i+1], f, indent=4)

    # saving npy files
    for i in range(0,len(npyfiles),2): 
        np.save(save_loc + npyfiles[i], npyfiles[i+1])
    
    # saving txtfiles
    for i in range(0,len(txtfiles),2):
         with open(save_loc + txtfiles[i], 'w') as f:
              for line in txtfiles[i+1]:
                   f.write(str(line))
                   f.write("\n")

    print("sim files saved under: " + save_loc)

def save_npy_list(sim_name, npyfiles, dir):

    save_loc = "./sims/" + sim_name + "/" + dir +"/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    for i in range(0,len(npyfiles),2): 
        np.save(save_loc + npyfiles[i], npyfiles[i+1])

    print("vars files saved under: " + save_loc )

def path_anims(sim_name, anim_file):
    save_loc = "./sims/" + sim_name + "/anims/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    return save_loc + anim_file

def path_plots(sim_name, plot_file):
    save_loc = "./sims/" + sim_name + "/anims/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    return save_loc + plot_file
    
# -------------------------------------


# POST PROCESSING
# -------------------------------------

def load_check(dir):
    print("loading data from /sim/ directory. ")
    name = input('please give the name of the /' + str(dir) + "/ folder: ")
    return name

def load_sim(sim_name, printt):
    path = "./sims/"
    load_loc = path + sim_name + "/"
    if printt == True:
        print("loading files from " + load_loc)
        specs = np.load(load_loc + "specs.npy")
        version = np.load(load_loc + "version.npy")
        print("-------------")
        print(specs)
        print("version: ", version)
        print("-------------")

    with open(load_loc+"param_dict.json", 'r') as f:
        param_dict = json.load(f)
    with open(load_loc+"post_dict.json", 'r') as f:
        post_dict = json.load(f)

    x = np.load(load_loc + "x.npy")
    t_arr = np.load(load_loc + "t_arr.npy")
    qtx = np.load(load_loc + "qtx.npy")

    return param_dict, post_dict, x, t_arr, qtx

def load_vars(vars, sim_name, print):
    path = "./sims/"
    load_loc_sims = path + sim_name + "/"
    load_loc = path + sim_name + "/vars/"
    if print == True:
        print("loading files from " + load_loc_sims)
        specs = np.load(load_loc_sims + "specs.npy")
        version = np.load(load_loc_sims + "version.npy")
        print("-------------")
        print(specs)
        print("version: ", version)
        print("-------------")

    var_files = ["T00(t,x).npy", "T0x(t,x).npy", "Txx(t,x).npy", "ep(t,x).npy", \
            "depdx(t,x).npy", "v(t,x).npy", "dvdx(t,x).npy", "P(t,x).npy", "gamma(t,x).npy"]
    
    T00_tx = np.load(load_loc + var_files[0])
    T0x_tx = np.load(load_loc + var_files[1])
    Txx_tx = np.load(load_loc + var_files[2])
    eptx = np.load(load_loc + var_files[3])
    depdx_tx = np.load(load_loc + var_files[4])
    vtx = np.load(load_loc + var_files[5])
    dvdx_tx = np.load(load_loc + var_files[6])
    Ptx = np.load(load_loc + var_files[7])
    gammatx = np.load(load_loc + var_files[8])


    vars_animate = [r"$T^{00}$ (GeV$^4$)", T00_tx, r"$T^{0x}$ (GeV$^4$)", T0x_tx, r"$T^{xx}$ (GeV$^4$)", Txx_tx, r"$\epsilon$ (GeV$^4$)", eptx, \
            r"$\partial_x \epsilon $", depdx_tx, r"$v $", vtx, r"$\partial_x v $", dvdx_tx, r"$P$ (GeV$^4$)", Ptx, \
                r"$\gamma$", gammatx]
    
    # var_names = ["T00(t,x)", 0, "T0x(t,x)", 0, "Txx(t,x)", 0, "ep(t,x)",0, \
    #         "depdx(t,x)", 0, "v(t,x)", 0, "dvdx(t,x)", 0, "P(t,x)"]
    var_names = vars
    
    return vars_animate, var_names

def load_vars2(sim_name, print):
    path = "./sims/"
    load_loc_sims = path + sim_name + "/"
    load_loc = path + sim_name + "/vars/"
    if print == True:
        print("loading files from " + load_loc_sims)
        specs = np.load(load_loc_sims + "specs.npy")
        version = np.load(load_loc_sims + "version.npy")
        print("-------------")
        print(specs)
        print("version: ", version)
        print("-------------")

    var_files = ["T00(t,x).npy", "T0x(t,x).npy", "Txx(t,x).npy", "ep(t,x).npy", \
            "depdx(t,x).npy", "v(t,x).npy", "dvdx(t,x).npy", "P(t,x).npy", "gamma(t,x).npy"]
    
    T00_tx = np.load(load_loc + var_files[0])
    T0x_tx = np.load(load_loc + var_files[1])
    Txx_tx = np.load(load_loc + var_files[2])
    eptx = np.load(load_loc + var_files[3])
    depdx_tx = np.load(load_loc + var_files[4])
    vtx = np.load(load_loc + var_files[5])
    dvdx_tx = np.load(load_loc + var_files[6])
    Ptx = np.load(load_loc + var_files[7])
    gammatx = np.load(load_loc + var_files[8])


    vars_animate = [r"$T^{00}$ (GeV$^4$)", T00_tx, r"$T^{0x}$ (GeV$^4$)", T0x_tx, r"$T^{xx}$ (GeV$^4$)", Txx_tx, r"$\epsilon$ (GeV$^4$)", eptx, \
            r"$\partial_x \epsilon $", depdx_tx, r"$v $", vtx, r"$\partial_x v $", dvdx_tx, r"$P$ (GeV$^4$)", Ptx, \
                r"$\gamma$", gammatx]
    
    var_names = ["T00(t,x)", 0, "T0x(t,x)", 0, "Txx(t,x)", 0, "ep(t,x)",0, \
            "depdx(t,x)", 0, "v(t,x)", 0, "dvdx(t,x)", 0, "P(t,x)"]

    
    return vars_animate, var_names
    
# -------------------------------------