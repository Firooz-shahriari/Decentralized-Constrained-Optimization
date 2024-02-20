import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np
import os
import glob
import copy as cp


#### put the dimensions that exists the OT_results folder 
#### if the folder does not exist: run OT_Time.py --n_source=dim --n_target=dim --n_sims=100, for different values of dim. 
dim = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]


avg_time    = []
output_path = "OT_results"

for d in dim:
    T = [[],[],[],[],[]]
    folder_path = glob.glob(os.path.join(output_path, f'Sim_dim_{d}*'))
    n_sim = 0
    for fld_path in folder_path:
        t_path = os.path.join(fld_path, 'Time.npy')
        tmp    = np.load(t_path)
        n_sim += tmp.shape[1]
        time_sum = np.sum(tmp,axis=1)
    avg = (time_sum/n_sim)
    each_node_avg = cp.copy(avg)
    for i in range(1,5):
        each_node_avg[i] = avg[i]/(2**(i+2))
    avg_time.append(each_node_avg)

tim = np.array(avg_time)

matplotlib.rcParams['text.usetex'] = True
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.family"] = "Arial"
font = FontProperties()
font.set_size(17)
font2 = FontProperties()
font2.set_size(17)
mark_every = 2
linewidth = 2

plt.figure(1, figsize=(8, 5))
plt.plot(dim, tim[:,0],   '-oy', markevery = mark_every, linewidth = linewidth, label= r'\textbf{LP-Solver}')
plt.plot(dim, tim[:,1],   '-sm', markevery = mark_every, linewidth = linewidth, label= r'\textbf{DAGP, $M=8$}')
plt.plot(dim, tim[:,2],   '-vr', markevery = mark_every, linewidth = linewidth, label= r'\textbf{DAGP, $M=16$}')
plt.plot(dim, tim[:,3],   '-*g', markevery = mark_every, linewidth = linewidth, label= r'\textbf{DAGP, $M=32$}')
plt.plot(dim, tim[:,4],   '-xb', markevery = mark_every, linewidth = linewidth, label= r'\textbf{DAGP, $M=64$}')


plt.grid(True)
plt.tick_params(labelsize='large', width=3)
plt.tick_params(labelsize=16, width=3)
plt.xlabel(r' \textbf{Problem size $(n)$}', fontproperties=font)
plt.ylabel(r'\textbf{Running Time} (in seconds)', fontproperties=font2)
plt.legend( prop={'size': 15})
path = os.path.join(output_path, 'Time_main')
plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
plt.show()
