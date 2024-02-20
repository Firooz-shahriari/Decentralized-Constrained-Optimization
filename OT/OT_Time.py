import argparse, os, time
import numpy as np
from Utils import Initialization, dagp_OT, lin_prog, save_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   type=str,   default="./",   help="Base_Directory")
    parser.add_argument("--n_source",   type=int,   default=4,      help="Number of Sources")
    parser.add_argument("--n_target",   type=int,   default=4,      help="Number of Targets")
    parser.add_argument("--eps_lp",     type=float, default=1e-4,   help="Change equality into inequality for linear programming")
    parser.add_argument("--eps1_dagp",  type=float, default=1e-7,   help="check if the algorithm has converged, the obejective value change between two successive iterations is lees than this value. ")
    parser.add_argument("--eps2_dagp",  type=float, default=1e-3,   help="how much we are close to the equality constraints")
    parser.add_argument("--step_dagp",  type=float, default=0.05,   help="step size of DAGP")
    parser.add_argument("--rho",        type=float, default=0.1,    help="rho parameter of DAGP")
    parser.add_argument("--alpha",      type=float, default=0.45,   help="alpha parameter of DAGP")
    parser.add_argument("--n_sims",     type=int,   default=2,      help="number of simulations to report the average running-time")
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    ns, nt, mu, nu, D, eps_lp, eps1_dagp, eps2_dagp, rho, alpha, step_size, init, n_sims, T_dagp_08, T_dagp_16, T_dagp_32, T_dagp_64, T_lp, iter_dagp_08, iter_dagp_16, iter_dagp_32, iter_dagp_64   = Initialization(
        n_source  = args.n_source,
        n_target  = args.n_target,
        eps_lp    = args.eps_lp,
        eps1_dagp = args.eps1_dagp,
        eps2_dagp = args.eps2_dagp,
        step_dagp = args.step_dagp,
        rho       = args.rho,
        alpha     = args.alpha,
        n_sims    = args.n_sims
    )

    for k in range(n_sims):
        print(f'\n\nSimulation: {k}, n: {ns}')
        X_lp, t_lp   = lin_prog(D,mu,nu,eps=eps_lp)
        print(f'CVXOPT_lp finished.    elapsed time:     {t_lp}')
        X_dagp_08, n_iter_08, t_dagp_08 = dagp_OT(8, step_size,D,mu,nu,rho,alpha,init=init,eps=eps1_dagp, eps_lp =eps2_dagp)
        print(f'DAGP_08 finished.      elapsed time:     {t_dagp_08}')
        X_dagp_16, n_iter_16, t_dagp_16 = dagp_OT(16,step_size,D,mu,nu,rho,alpha,init=init,eps=eps1_dagp, eps_lp =eps2_dagp)
        print(f'DAGP_16 finished.      elapsed time:     {t_dagp_16}')
        X_dagp_32, n_iter_32, t_dagp_32 = dagp_OT(32,step_size,D,mu,nu,rho,alpha,init=init,eps=eps1_dagp, eps_lp =eps2_dagp)
        print(f'DAGP_32 finished.      elapsed time:     {t_dagp_32}')
        X_dagp_64, n_iter_64, t_dagp_64 = dagp_OT(64,step_size,D,mu,nu,rho,alpha,init=init,eps=eps1_dagp, eps_lp =eps2_dagp)
        print(f'DAGP_64 finished.      elapsed time:     {t_dagp_64}')
        T_lp.append(t_lp)
        T_dagp_08.append(t_dagp_08)
        iter_dagp_08.append(n_iter_08)
        T_dagp_16.append(t_dagp_16)
        iter_dagp_16.append(n_iter_16)
        T_dagp_32.append(t_dagp_32)
        iter_dagp_32.append(n_iter_32)
        T_dagp_64.append(t_dagp_64)
        iter_dagp_64.append(n_iter_64)

    sim_time = str(time.time()).split(".")[0]
    folder   = 'Sim' + '_dim_' + str(ns) +  '_Time_' + sim_time
    save_dir = os.path.join(args.base_dir, "OT_results", folder)
    os.makedirs(save_dir)

    save_config(os.path.join(save_dir, 'Config.txt'), ns, nt, eps_lp, eps1_dagp, eps2_dagp, step_size, rho, alpha, n_sims)

    T = np.zeros((5,n_sims))
    N_iter = np.zeros((5,len(iter_dagp_08)))
    T[0] = np.array(T_lp)
    T[1] = np.array(T_dagp_08)
    T[2] = np.array(T_dagp_16)
    T[3] = np.array(T_dagp_32)
    T[4] = np.array(T_dagp_64)
    N_iter[0] = np.ones(len(iter_dagp_08))
    N_iter[1] = np.array(iter_dagp_08)
    N_iter[2] = np.array(iter_dagp_16)
    N_iter[3] = np.array(iter_dagp_32)
    N_iter[4] = np.array(iter_dagp_64)

    np.save(os.path.join(save_dir, 'Time.npy'), T)
    np.save(os.path.join(save_dir, 'Num_iter.npy'), N_iter)


if __name__ == "__main__":
    main()