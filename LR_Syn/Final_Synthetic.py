import numpy as np
from graph.graph import Random
from utilities.utilities import sim_params_synthetic, save_plot_syn
from analysis.analysis import error
from Problems.synthetic_cosh import synthetic
from Optimizers import DOPTIMIZER as dopt
import os

#### simulation parameters
seed, depoch, num_nodes, dim, edge_prob, ldf, graph_type, theta_0, step_size_DAGP, rho_DAGP, alpha_DAGP, eps_DDPS, p_DDPS = sim_params_synthetic(\
    seed                   = 34,  
    decentralized_epochs   = 10000,
    num_nodes              = 10,
    dimension              = 20, 
    edge_probability       = 1.0, 
    divinding_factor       = 2,
    graph_type             = 'Random',
    step_size_DAGP         = 0.05,   
    rho_DAGP               = 1e-2,            
    alpha_DAGP             = 0.1,
    eps_DDPS               = 0.05,
    p_DDPS                 = 0.5
    )

#### Problem setup: parameters of the synthetic functions and constraints. 
prd = synthetic(seed, num_nodes, dim)
error_prd = error(prd,np.zeros(num_nodes),0)


#### Create gossip matrices
zero_row_sum, zero_column_sum, row_stochastic, column_stochastic = Random(num_nodes, edge_prob, ldf).directed()


#### Run the optimization algorithms and compute the performance metrics
theta_ddps = dopt.DDPS(\
    prd, row_stochastic, column_stochastic, p_DDPS, int(depoch),theta_0, eps_DDPS)
theta_DAGP,_, h_itrs, g_itrs = dopt.DAGP(\
    prd, zero_row_sum, zero_column_sum, step_size_DAGP, int(depoch), theta_0, rho_DAGP, alpha_DAGP, cons = True)

res_F_ddps = error_prd.cost_gap_path(np.sum(theta_ddps, axis=1)/num_nodes)
res_F_DAGP = error_prd.cost_path(np.sum(theta_DAGP, axis=1)/num_nodes)

fesgp_ddps = error_prd.feasibility_gap_syn(np.sum(theta_ddps, axis=1)/num_nodes)
fesgp_DAGP = error_prd.feasibility_gap_syn(np.sum(theta_DAGP, axis=1)/num_nodes)


#### save data and plot results
save_plot_syn(dim, num_nodes, seed, res_F_DAGP, fesgp_DAGP, theta_DAGP, g_itrs, h_itrs, res_F_ddps,\
               fesgp_ddps, theta_ddps, step_size_DAGP, eps_DDPS, p_DDPS, rho_DAGP, alpha_DAGP, \
                 depoch, graph_type, edge_prob, ldf, row_stochastic,current_dir=os.path.dirname(os.path.abspath(__file__)), save_results_folder = 'plots_synthetic')
    
