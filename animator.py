import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import PillowWriter
plt.rcParams['font.size'] = 13 
plt.rcParams["figure.figsize"] = (9,5)

import data_manager as data_m

def animate(sim_name, var_name, var_label, var_data):

    linestyles1 = ['k-.', 'r', 'b', 'g', 'm', 'm:', 'y:']
    linestyles2 = ['k', 'r-.', 'b--', 'g:', 'm-.', 'm-.', 'y:']

    fig,ax = plt.subplots()

    metadata = dict(title = 'Movie', artist = 'nclar')
    writer = PillowWriter(fps = 15, metadata = metadata) # fps here

    param_dict, post_dict, x, t_arr, qtx = data_m.load_sim(sim_name, False)

    print("making animations...")

    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"] 

    anim = var_data
    ylabel = var_label
    ylabel0 = ylabel + " (t=0)"

    with writer.saving(fig, data_m.path_anims(sim_name, var_name+".gif"), dpi=100):
            
        for i in range(0, len(t_arr)): # play with last entry to skip time, speed up animations

            ax.clear()
            ax.plot(x, anim[0], linestyles1[0], label = ylabel0)
            ax.plot(x, anim[i], linestyles1[2], label = ylabel)   

            title = r"$t($1/GeV$) = $" + str('%.2f'%(t_arr[i]))

            ax.set_xlabel(r"$x$ " + "(1/GeV)")
            ax.set_ylabel(ylabel)
            ax.set_xlim([x[0], x[-1]])

            # adapative y boundaries
            if np.min(anim[:]) < 0:
                ymin = (np.min(anim[:]) + 0.2 * np.min(anim[:]) )
            if np.min(anim[:]) > 0:
                ymin = (np.min(anim[:]) - 0.2 * np.min(anim[:]) )
            if np.max(anim[:]) > 0:
                ymax = (np.max(anim[:]) + 0.2 * np.max(anim[:]))
            if np.max(anim[:]) <0:  
                ymax = (np.max(anim[:]) - 0.2 * np.max(anim[:]))

            if np.min(anim[:]) == 0:
                # ymin = (np.min(anim[:]) - 0.5 * np.abs(np.max(anim[:]) ))
                ymid = (np.abs(np.min(anim[:])) +  np.abs(np.max(anim[:]) ))/2
                ymin = np.min(anim[:]) - 0.2 *ymid

            if np.max(anim[:]) == 0:
                # ymax = (np.max(anim[:]) + 0.5 * np.abs(np.min(anim[:]) ))
                ymid = (np.abs(np.min(anim[:])) +  np.abs(np.max(anim[:]) ))/2
                ymax = np.max(anim[:]) + 0.2 * ymid



            ax.set_ylim( [ymin, ymax ] ) 

            ax.set_title(title)
            ax.legend()
            ax.legend(loc = 'upper right')

            writer.grab_frame()

            plt.tight_layout()

        plt.savefig(data_m.path_plots(sim_name, var_name+".pdf"))

        print("animation for " + str(var_name) + " completed.")

def animate_all(sim_name, var_names, vars):
    for i in range(0, len(vars), 2):
        animate(sim_name, var_names[i], vars[i], vars[i+1])