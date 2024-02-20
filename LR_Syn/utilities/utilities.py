########################################################################################################################
####---------------------------------------------------Utilities----------------------------------------------------####
########################################################################################################################
import os
import numpy as np
import matplotlib as mpl
import time
import os
import sys
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def monitor(name,current,total):
    if (current+1) % (total/10) == 0:
        print ( name + ' %d%% completed' % int(100*(current+1)/total) )

def Save_config_LR(path, seed, nodes, dimension, DAGP_step, ADDOPT_step, PushPull_step, Centralized_step, DAGP_rho, DAGP_alpha,\
                   Centralized_epochs, Decentralized_epochs, Num_training_samples, regularization_factor,\
                    Smothness_constant, Graph, Prob_edges, Laplacian_dividing_factor):
    readme = 'The configuration of this simulation: \n' + \
    'seed:                                 ' + str(seed) + '\n' + \
    'number of nodes:                      ' + str(nodes) + '\n' + \
    'dimension of data:                    ' + str(dimension) + '\n' + \
    'step size of DAGP:                    ' + str(DAGP_step)  + '\n' + \
    'step size of ADDOPT:                  ' + str(ADDOPT_step) + '\n' + \
    'step size of PushPull:                ' + str(PushPull_step) + '\n' + \
    'step size of CentralizedGP:           ' + str(Centralized_step) + '\n' + \
    'DAGP_rho:                             ' + str(DAGP_rho) + '\n' + \
    'DAGP_alpha:                           ' + str(DAGP_alpha) + '\n' + \
    'Centralized_epochs:                   ' + str(Centralized_epochs) + '\n' + \
    'Decentralized_epochs:                 ' + str(Decentralized_epochs) + '\n' + \
    'Num_training_samples:                 ' + str(Num_training_samples) + '\n' + \
    'regularization_factor:                ' + str(regularization_factor) + '\n' + \
    'Smothness_constant:                   ' + str(Smothness_constant) + '\n' + \
    'Graph:                                ' + Graph + '\n' + \
    'Prob_edges:                           ' + str(Prob_edges) + '\n' + \
    'Laplacian_dividing_factor:            ' + str(Laplacian_dividing_factor)
    with open(path, "w") as text_file:
        text_file.write(readme)

def Save_config_syn(path, seed, nodes, dimension, DAGP_step, DDPS_epsilon, Decay_DDPS, DAGP_rho, DAGP_alpha,\
                     Decentralized_epochs, Graph, Prob_edges, Laplacian_dividing_factor):

    readme = 'The configuration of this simulation: \n' + \
    'seed:                                 ' + str(seed) + '\n' + \
    'number of nodes:                      ' + str(nodes) + '\n' + \
    'dimension of data:                    ' + str(dimension) + '\n' + \
    'step size of DAGP:                    ' + str(DAGP_step)  + '\n' + \
    'DDPS Epsilon:                         ' + str(DDPS_epsilon)  + '\n' + \
    'step size of DDPS Decay factor:       ' + str(Decay_DDPS)  + '\n' + \
    'DAGP_rho:                             ' + str(DAGP_rho) + '\n' + \
    'DAGP_alpha:                           ' + str(DAGP_alpha) + '\n' + \
    'Decentralized_epochs:                 ' + str(Decentralized_epochs) + '\n' + \
    'Graph:                                ' + Graph + '\n' + \
    'Prob_edges:                           ' + str(Prob_edges) + '\n' + \
    'Laplacian_dividing_factor:            ' + str(Laplacian_dividing_factor)
    with open(path, "w") as text_file:
        text_file.write(readme)


def nx_options():
    options = {
     'node_color': 'skyblue',
     'node_size': 10,
     'edge_color': 'grey',
     'width': 0.5,
     'arrows': False,
     'node_shape': 'o',}
    return options

def sim_params_synthetic(seed,decentralized_epochs,num_nodes,dimension, edge_probability, divinding_factor,graph_type,step_size_DAGP,rho_DAGP,alpha_DAGP,eps_DDPS,p_DDPS):
    return seed,decentralized_epochs,num_nodes,dimension, edge_probability, divinding_factor,graph_type,np.random.randn(num_nodes,dimension),step_size_DAGP,rho_DAGP,alpha_DAGP,eps_DDPS,p_DDPS


def sim_params_LR(seed, num_nodes, num_train_data, centralized_epochs, decentralized_epochs, edge_prob , dividing_factor, step_size_addopt_factor, step_size_pp_factor, step_size_DAGP_factor, step_center_factor, rho_dagp, alpha_dagp, graph_type):
    np.random.seed(seed)
    theta_0 = np.random.normal(0,1,(num_nodes, 785))
    theta_c0 = np.random.normal(0,1, 785)
    lamdaa = 1/num_train_data
    return seed, num_nodes, num_train_data, lamdaa, centralized_epochs, decentralized_epochs, edge_prob, dividing_factor, theta_c0, theta_0, step_size_addopt_factor, step_size_pp_factor, step_size_DAGP_factor, step_center_factor, rho_dagp, alpha_dagp, graph_type


def save_plot_syn(dim, num_nodes, seed, res_F_DAGP, fesgp_DAGP, theta_DAGP, g_itrs, h_itrs, res_F_ddps,\
              fesgp_ddps, theta_ddps, step_size_DAGP, eps_DDPS, p_DDPS, rho_DAGP, alpha_DAGP, depoch, graph_type,\
                  edge_prob, ldf, row_stochastic, current_dir, save_results_folder):
    os.chdir(current_dir)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)
    os.chdir(save_results_folder)

    sim_time = str(time.time()).split(".")[0]
    folder   = 'SynLogCosh_n_' + str(dim) + '_N_' + str(num_nodes) + '_seed_' + str(seed) + '_Time_' + sim_time
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.savez(os.path.join(folder, 'DAGP'),  res_F=res_F_DAGP , fesgp=fesgp_DAGP, theta=theta_DAGP)
    np.savez(os.path.join(folder, 'DDPS'),  res_F=res_F_ddps , fesgp=fesgp_ddps, theta=theta_ddps)
    Save_config_syn( os.path.join(folder, 'Config.txt'), seed, num_nodes, dim, step_size_DAGP, eps_DDPS, p_DDPS, rho_DAGP, alpha_DAGP, depoch, graph_type, edge_prob, ldf)

    label_size   = 18
    legend_size  = 18
    linewidth    = 2.5
    linewidth2   = 1.2
    tick_label   = 20

    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2.2
    plt.rcParams["font.family"] = "Arial"

    font = FontProperties()
    font.set_size(label_size)
    font2 = FontProperties()
    font2.set_size(legend_size)
    mark_every = 5000

    plt.figure(1, figsize=(7, 4))
    plt.plot(res_F_DAGP,'-oy', markevery = mark_every, linewidth = linewidth)
    plt.plot(res_F_ddps,'-^b', markevery = mark_every, linewidth = linewidth)
    plt.grid(True)

    if num_nodes == 10:
        plt.yscale('log')

    plt.tick_params(labelsize=tick_label, width=3)
    plt.ylabel(r'\textbf{Objective Value}', fontproperties=font)
    plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
    plt.legend((r'\textbf{DAGP}', r'\textbf{DDPS}'), prop={'size': legend_size}) 
    filename = str(time.time()).split(".")[0]
    timestamp = np.array([int(filename)])
    path = os.path.join(folder, 'objective')
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
    plt.savefig( path + ".eps", format = 'eps', pad_inches=0.05, bbox_inches ='tight')
    plt.show(block = False)
    #------------------------------------------------------------------------------

    plt.figure(2, figsize=(7, 4))
    plt.plot(fesgp_DAGP,'-oy', markevery = mark_every, linewidth = linewidth)
    plt.plot(fesgp_ddps,'-^b', markevery = mark_every, linewidth = linewidth)
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize=tick_label, width=3)
    plt.ylabel(r'\textbf{Feasibility Gap}', fontproperties=font2)
    plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
    plt.legend((r'\textbf{DAGP}', r'\textbf{DDPS}'), prop={'size': legend_size}) 
    path = os.path.join(folder, 'feasibility')
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
    plt.savefig( path + ".eps", format = 'eps', pad_inches=0.05, bbox_inches ='tight')
    plt.show(block = False)
    #------------------------------------------------------------------------------
    G  = np.zeros((num_nodes, len(g_itrs)))
    H  = np.zeros((num_nodes, len(h_itrs)))
    GH = np.zeros((num_nodes, len(h_itrs)))
    X  = np.zeros((num_nodes, len(h_itrs)))
    DX = np.zeros((num_nodes, len(h_itrs)))
    gs = np.sum(g_itrs,axis = 1)/num_nodes
    Gs = np.zeros(len(h_itrs))
    for i in range(len(h_itrs)):
        Gs[i] = np.linalg.norm(gs[i])
        for j in range(num_nodes):
            #--------
            g_tmp     = g_itrs[i]
            #--------
            h_tmp    = h_itrs[i]
            #--------
            x_tmp    = theta_DAGP[i]
            #--------
            G[j][i]  = np.linalg.norm(g_tmp[j])
            H[j][i]  = np.linalg.norm(h_tmp[j])
            X[j][i]  = np.linalg.norm(x_tmp[j])
            DX[j][i] = np.linalg.norm(x_tmp[j] - x_tmp[0])
    np.savez(os.path.join(folder, 'DAGP_XHG'), X=X, H=H, G=G, GH=GH)
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    plt.figure(3, figsize=(7, 4))
    x0 = X[0]; x1 = X[1]; x2 = X[2]; x3 = X[3]; x4 = X[4]; x5 = X[5]

    ###### ||x_i - x_j|| l2 norm 
    plt.plot(DX[5], linewidth = linewidth2, label = r' $ \left \| {\bf{x}}^\nu  -  {\bf{x}}^0 \right \|_2$' )
    plt.plot(DX[1], linewidth = linewidth2)
    plt.plot(DX[2], linewidth = linewidth2)
    plt.plot(DX[3], linewidth = linewidth2)
    plt.plot(DX[4], linewidth = linewidth2)
    plt.yscale('log')


    plt.grid(True)
    plt.tick_params(labelsize=tick_label, width=3)
    # plt.ylabel(r' $ \left \| {\bf{x}}^\nu \right \|_2 $', fontproperties=font)
    plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
    # legend1 = plt.legend(('Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5'), loc = "lower right", prop={'size': legend_size2})
    # plt.gca().add_artist(legend1)
    # legend2 = plt.legend(handlelength = 0.0, handletextpad = 0.0, markerscale = 0.0, loc = 1, prop={'size': legend_size+3})
    # plt.gca().add_artist(legend2)
    plt.legend(handlelength = 0.0, handletextpad = 0.0, markerscale = 0.0, loc = 1, prop={'size': legend_size+3})
    path = os.path.join(folder, 'iterates')
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
    plt.savefig( path + ".eps", format = 'eps', pad_inches=0.05, bbox_inches ='tight')
    plt.show(block=False)

    #------------------------------------------------------------------------------

    plt.figure(4, figsize=(7, 4))
    plt.plot(Gs,'-y', linewidth = linewidth, label = r'$\left\| \sum_{v \in \mathcal{V}} {\bf{g}}^\nu  \right \|_2$')
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize=tick_label, width=3)
    plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
    plt.legend(handlelength = 0.0, handletextpad = 0.0 , markerscale = 0.0, loc = 1, prop={'size': legend_size+3})
    path = os.path.join(folder, 'sumg')
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
    plt.savefig( path + ".eps", format = 'eps', pad_inches=0.05, bbox_inches ='tight')
    plt.show(block=False)

    #------------------------------------------------------------------------------
    plt.figure(5)
    G = nx.from_numpy_matrix(np.matrix(row_stochastic), create_using=nx.DiGraph)
    layout = nx.circular_layout(G)
    nx.draw(G, layout)

    path = os.path.join(folder, 'graph')
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
    plt.savefig( path + ".eps", format = 'eps', pad_inches=0.05, bbox_inches ='tight')
    #------------------------------------------------------------------------------


def save_plot_LR(prd, res_F_DAGP, res_F_ADDOPT, res_F_pp, seed, num_nodes, cepoch, depoch, edge_prob, ldf, step_size_addopt_factor, step_size_pp_factor, step_size_DAGP_factor, step_center_factor, rho_dagp, alpha_dagp, graph_type, row_stochastic, current_dir, save_results_folder = 'plots_LR'):
                
    os.chdir(current_dir)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)
    os.chdir(save_results_folder)

    sim_time = str(time.time()).split(".")[0]
    folder   = 'LR_MNIST_n_' + str(num_nodes) + '_N_' + str(prd.N ) + '_Time_' + sim_time
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(os.path.join(folder, 'DAGP'     ),  res_F_DAGP)
    np.save(os.path.join(folder, 'ADDOPT'   ),  res_F_ADDOPT)
    np.save(os.path.join(folder, 'PushPull' ),  res_F_pp)
    Save_config_LR( os.path.join(folder, 'Config.txt'), seed, num_nodes, prd.p, step_size_DAGP_factor/prd.L, \
                step_size_addopt_factor/prd.L, step_size_pp_factor/prd.L, step_center_factor/prd.L, \
                    rho_dagp, alpha_dagp, cepoch, depoch, prd.N, prd.L, prd.L, graph_type, edge_prob, ldf)

    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    font = FontProperties()
    font.set_size(17)
    mark_every = 5000
    linewidth = 2

    plt.figure(1, figsize=(7, 4))
    plt.plot(res_F_DAGP,   '-oy', markevery = mark_every,linewidth = linewidth)
    plt.plot(res_F_ADDOPT, '-sm', markevery = mark_every,linewidth = linewidth)
    plt.plot(res_F_pp,     '-vr', markevery = mark_every,linewidth = linewidth)

    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize='large', width=3)
    plt.tick_params(labelsize=17, width=3)
    plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
    plt.ylabel(r'\textbf{Optimality Gap}', fontproperties= font)
    plt.legend((r'\textbf{DAGP}', r'\textbf{ADD-OPT}', r'\textbf{Push-Pull}'), prop={'size': 16})
    path = os.path.join(folder, 'objective')
    plt.show(block=False)
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    plt.figure(5)
    G = nx.from_numpy_matrix(np.matrix(row_stochastic), create_using=nx.DiGraph)
    layout = nx.circular_layout(G)
    nx.draw(G, layout)
    path = os.path.join(folder, 'graph_LR')
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
    #------------------------------------------------------------------------------

