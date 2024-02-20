
import numpy as np
from graph.graph import Random
from utilities.utilities import sim_params_LR, save_plot_LR
from analysis.analysis import error
from Problems.logistic_regression import LR_L2
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
import matplotlib as mpl
import os


#### simulation parameters
seed, num_nodes, num_train_data, lamdaa, cepoch, depoch, edge_prob, ldf, theta_c0, theta_0, step_size_addopt_factor, step_size_pp_factor, step_size_DAGP_factor, step_center_factor, rho_dagp, alpha_dagp, graph_type = sim_params_LR(\
    seed = 4,
    num_nodes = 20,
    num_train_data =  10000,
    centralized_epochs = 20000,
    decentralized_epochs = 4000,
    edge_prob = 0.3,
    dividing_factor = 2,
    step_size_addopt_factor = 1.5,
    step_size_pp_factor = 2,
    step_size_DAGP_factor = 2,
    step_center_factor = 0.5,
    rho_dagp = 1e-1,
    alpha_dagp = 0.1,
    graph_type = 'Random'
    )


#### Logistic Regression Problem ==> p: dimension of the model,   L: L-smooth constant,  N: total number of training samples,  b: average number of local samples,  l: regularization factor  
os.chdir(os.path.dirname(os.path.abspath(__file__)))                                           
lr_0 = LR_L2(num_nodes, limited_labels=False, balanced=True, train=num_train_data, regularization=True, lamda=lamdaa)  ## instantiate the problem class


#### Create gossip matrices
zero_row_sum, zero_column_sum, row_stochastic, column_stochastic = Random(num_nodes, edge_prob, ldf).directed()


## find the optimal solution
_, theta_opt, F_opt = copt.CGD(lr_0, step_center_factor/lr_0.L, cepoch, theta_c0)
error_prd = error(lr_0,theta_opt,F_opt)


#### Run the optimization algorithms and compute the performance metrics
theta_DAGP,_,_,_ = dopt.DAGP(lr_0, zero_row_sum , zero_column_sum  , step_size_DAGP_factor/lr_0.L, int(depoch), theta_0, rho_dagp , alpha_dagp, cons=False)
theta_ADDOPT     = dopt.ADDOPT(lr_0, row_stochastic, column_stochastic, step_size_addopt_factor/lr_0.L, int(depoch), theta_0)
theta_pp         = dopt.PushPull(lr_0, row_stochastic, column_stochastic , step_size_pp_factor/lr_0.L, int(depoch), theta_0)

res_F_pp         = error_prd.cost_gap_path(np.sum(theta_pp,     axis=1)/num_nodes)
res_F_ADDOPT     = error_prd.cost_gap_path(np.sum(theta_ADDOPT, axis=1)/num_nodes)
res_F_DAGP       = error_prd.cost_gap_path(np.sum(theta_DAGP,   axis=1)/num_nodes)


#### save data and plot results
save_plot_LR(lr_0, res_F_DAGP, res_F_ADDOPT, res_F_pp, seed, num_nodes, cepoch, depoch, edge_prob, ldf, step_size_addopt_factor, step_size_pp_factor, step_size_DAGP_factor, step_center_factor, rho_dagp, alpha_dagp, graph_type, row_stochastic, current_dir=os.path.dirname(os.path.abspath(__file__)), save_results_folder = 'plots_LR')
    
