import ot
import tensorly
import copy as cp
import os
from cvxopt import matrix, solvers
import time
import networkx as nx
import numpy as np



class Random_graph:
    def __init__(self,number_of_nodes, prob, Laplacian_dividing_factor_W, Laplacian_dividing_factor_Q):
        self.size = number_of_nodes
        self.prob = prob
        self.LDF_W  = Laplacian_dividing_factor_W
        self.LDF_Q  = Laplacian_dividing_factor_Q

    def directed(self):     # I should think more about how I am generating them. num_edges= N(N-1)
        nn = self.size*(self.size)
        indices = np.arange(nn)
        np.random.shuffle(indices)
        nonz = int(np.floor(nn*self.prob))
        ind = indices[:nonz]
        Z = np.zeros(nn)
        Z[ind] = 1.0
        D = Z.reshape(self.size,self.size)

        for i in range(self.size):
            if D[i][i] == 1.:
                D[i][i] = 0.

        GG = nx.from_numpy_matrix(np.matrix(D), create_using=nx.DiGraph)
        largest = max(nx.kosaraju_strongly_connected_components(GG), key=len)

        adj = np.zeros((len(largest), len(largest)))
        v = 0
        w = 0
        for i in largest:
            for j in largest:
                adj[v][w] = D[i][j]
                w +=1
            w = 0
            v +=1
        row_sum = np.sum(adj, axis = 1)
        col_sum = np.sum(adj, axis = 0)
        l_in  = np.diag(row_sum) - adj
        l_out = np.diag(col_sum) - adj
        ZR  = l_in  / (self.LDF_W*np.max(row_sum))
        ZC  = l_out / (self.LDF_Q*np.max(col_sum))
        RS  = np.eye(self.size) - ZR
        CS  = np.eye(self.size) - ZC
        return ZR, ZC, RS, CS

def distribute(m,n,M):   
    l  = np.ceil( (m+n)/M )
    a1 = np.ceil( m/l )
    a2 = np.ceil( n/l )

    if a1 == min(a1,a2):
        a_1 = min(a1,a2)
        a_2 = M-a_1
    elif a2 == min(a1,a2):
        a_2 = min(a1,a2)
        a_1 = M-a_2 
    l_1 = np.ceil( m/a_1 )
    l_2 = np.ceil( n/a_2 )

    row_ind = []
    col_ind = [] 
    for i in range(M):
        row_ind.append([j+l_1*i       for j in range(int(l_1)) if i< a_1 and j+l_1*i<m       ])
        col_ind.append([j+l_2*(i-a_1) for j in range(int(l_2)) if i>=a_1 and j+l_2*(i-a_1)<n ])
    return row_ind, col_ind, a_1

def gaussian(n, mean, sigma):
    x = np.arange(n)/(n-1)
    g = np.zeros(n)
    for i in range(n):
        g[i] = ( np.exp(-(x[i] - mean) ** 2 / (2 * sigma ** 2)) ) / ( np.sqrt( 2*np.pi*sigma**2) )
    return g/g.sum()

def simplex_projection(y, a):
    u = np.flip(np.sort(y))
    k = 0
    while k <= len(y)-1:
        tau = (np.sum(u[:k+1]) - a)/(k+1) 
        if tau <= u[k]:
            k += 1
        else:
            break
    tau = (np.sum(u[:k]) - a)/(k) 
    out = y -tau
    out[out < 0] = 0 
    return out

def network_grad(D, row_ind, col_ind, nr):
    nN = len(row_ind)
    ns,nt = D.shape
    f_grad = np.zeros((nN, ns, nt))

    for i in range(nN):
        rows = np.asarray(row_ind[i]).astype(int)
        cols = np.asarray(col_ind[i]).astype(int)
        if i < nr:
            f_grad[i,rows,:] = D[rows,:] 
        else:
            f_grad[i,:,cols] = np.transpose(D[:,cols])
    return f_grad/2

def Initialization(n_source, n_target, eps_lp, eps1_dagp, eps2_dagp, step_dagp, rho, alpha, n_sims):
    ns = n_source
    nt = n_target
    eps_lp    = eps_lp         # relaxing equailty constraints with two inequality
    eps1_dagp = eps1_dagp      # objective value difference computed at two successive iterations, to be sure the algorithm has converged.
    eps2_dagp = eps2_dagp      # another condition for stopping dagp, the constraint should be satisfied, we will relax them with epsilon. 


    ms = 1/3   # mean source distribution
    mt = 2/3   # mean target distribution
    ss = 1/4   # std  source distribution
    st = 1/8   # std  target dostribution

    init      = 'random'
    step_size = step_dagp
    rho       = rho
    alpha     = alpha


    mu       = gaussian(ns,ms,ss)
    nu       = gaussian(nt,mt,st)
    ns_range = np.arange(ns)/(ns-1)
    nt_range = np.arange(nt)/(nt-1)
    D        = ot.dist(ns_range.reshape(ns,1), nt_range.reshape(nt,1))
    n_sims   = n_sims

    T_dagp_08    = []
    T_dagp_16    = []
    T_dagp_32    = []
    T_dagp_64    = []
    T_lp         = []
    iter_dagp_08 = []
    iter_dagp_16 = []
    iter_dagp_32 = []
    iter_dagp_64 = []

    return ns, nt, mu, nu, D, eps_lp, eps1_dagp, eps2_dagp, rho, alpha, step_size, init, n_sims,T_dagp_08,T_dagp_16,T_dagp_32,T_dagp_64,T_lp,iter_dagp_08,iter_dagp_16,iter_dagp_32,iter_dagp_64

def dagp_OT(nN,step_size,D,mu,nu,rho,alpha,edge_prob=1.0,init='random',eps=1e-7, eps_lp=1e-5):

    edge_prob = edge_prob
    ldf_W     = 1.2
    ldf_Q     = 1.2

    W, Q,_,_ = Random_graph(nN, edge_prob, ldf_W, ldf_Q).directed()

    ns, nt = D.shape
    row_ind, col_ind, nr = distribute(ns,nt,nN)
    if init == 'random':
        X = np.random.rand(nN,ns,nt)
        for i in range(nN):
            tmp = np.random.rand(ns,nt)
            tmp = tmp / np.sum(tmp)
            X[i,:,:] = tmp
    elif init == 'zeros':
        X = np.zeros((nN,ns,nt))

    f_grad = network_grad(D,row_ind, col_ind, nr)
    G  = np.zeros((nN,ns,nt))
    H  = np.zeros((nN,ns,nt))
    Z  = np.zeros((nN,ns,nt))
    nH = np.zeros((nN,ns,nt))

    n_iter = 0
    f_pre  = 1e6
    f_current = 1e5

    cond = True
    t_dagp = 0
    
    while cond:
        start = time.time()
        Z = X - tensorly.tenalg.mode_dot(X,W,0) - step_size*(f_grad-G)
        temp = cp.deepcopy(Z)
        for i in range(nN):
            rows = np.asarray(row_ind[i]).astype(int)
            cols = np.asarray(col_ind[i]).astype(int)
            if i < nr:
                for j in rows:
                    temp[i,j,:] = simplex_projection(Z[i,j,:], mu[j])  
            else:
                for j in cols:
                    temp[i,:,j] = simplex_projection(Z[i,:,j], nu[j])
        X = cp.deepcopy(temp)

        nH = H - tensorly.tenalg.mode_dot(H-G,Q,0) 
        G  = G + rho*( f_grad-G + (Z-X)/step_size ) + alpha*( H-G )
        H  = nH
        t_dagp += time.time()-start
        n_iter+=1
        f_pre = cp.deepcopy(f_current)
        f_current = np.sum(D*np.sum(X,axis=0)/nN)

        if np.abs(f_pre - f_current) > eps:
            cond == True
        else: 
            if n_iter%100 == 0:
                tmp1 = np.linalg.norm( np.sum(np.sum(X,axis=0)/nN, axis=1) - mu ) > eps_lp 
                if tmp1 == True:
                    cond = True
                else: 
                    tmp2 = np.linalg.norm( np.sum(np.sum(X,axis=0)/nN, axis=0) - nu ) > eps_lp
                    cond = tmp2

    return X, n_iter, t_dagp


def dagp_OT_DA(D, nN, edge_prob, step_size, rho, alpha, ldf_W, ldf_Q, max_iter, eps, eps_lp):
    ns = D.shape[0]
    nt = D.shape[1]
    mu = ot.unif(ns)
    nu = ot.unif(nt)
    row_ind, col_ind, nr = distribute(ns, nt, nN)
    W, Q,_,_ = Random_graph(nN, edge_prob, ldf_W, ldf_Q).directed()

    G = np.zeros((nN,ns,nt))
    H = np.zeros((nN,ns,nt))
    X = np.zeros((nN,ns,nt))
    for i in range(nN):
        tmp = np.random.rand(ns,nt)
        tmp = tmp / np.sum(tmp)
        X[i,:,:] = tmp

    f_grad = network_grad(D,row_ind, col_ind, nr)

    f_pre = 1e6
    f_current = 1e5
    cond = True
    n_iter = 0

    while cond:
        print(f'iteration: {n_iter+1}')
        Z = X - tensorly.tenalg.mode_dot(X,W,0) - step_size*(f_grad-G)
        temp = cp.deepcopy(Z)
        for i in range(nN):
            rows = np.asarray(row_ind[i]).astype(int)
            cols = np.asarray(col_ind[i]).astype(int)
            if i < nr:
                for j in rows:
                    temp[i,j,:] = simplex_projection(Z[i,j,:], mu[j])  
            else:
                for j in cols:
                    temp[i,:,j] = simplex_projection(Z[i,:,j], nu[j])
        X = cp.deepcopy(temp)

        nH = H - tensorly.tenalg.mode_dot(H-G,Q,0) 
        G  = G + rho*( f_grad-G + (Z-X)/step_size ) + alpha*( H-G )
        H  = nH
        n_iter+=1
        f_pre = cp.deepcopy(f_current)
        f_current = np.sum(D*np.sum(X, axis=0)/nN)
        if np.abs(f_pre - f_current) > eps:
            cond == True
        else: 
            if n_iter%1 == 0:
                check_mu = np.linalg.norm(np.sum(np.sum(X, axis=0)/nN, axis=1) - mu )
                tmp1 = check_mu > eps_lp 
                if tmp1 == True:
                    cond = True
                else: 
                    check_nu = np.linalg.norm(np.sum(np.sum(X, axis=0)/nN, axis=0) - nu )
                    tmp2 = check_nu > eps_lp
                    cond = tmp2

        if n_iter == max_iter:
            cond = False

    plan = np.sum(X, axis=0)/nN
    return plan

def lin_prog(D,mu,nu,eps=1e-9):
    ns, nt = D.shape
    c = matrix(D.ravel())
    G = matrix(np.diag(-1*np.ones(D.shape[0]*D.shape[1])))
    h = matrix(np.zeros(D.shape[0]*D.shape[1]))

    ns,nt = D.shape
    t_kron = np.ones(nt)
    T_kron = np.diag(np.ones(ns))
    A_1 = np.kron(T_kron,t_kron)
    A_1 = np.transpose(A_1)

    A_2 = np.zeros((ns*nt, nt))
    for j in range(nt):
        ind = [j+i*nt for i in range(ns)]
        A_2[ind,j] = 1

    # add epsilon to change equality constriants to inequality
    G = matrix(np.transpose(np.concatenate((np.diag(-1*np.ones(D.shape[0]*D.shape[1])), np.concatenate((A_1,A_2),axis=1), -1*np.concatenate((A_1,A_2),axis=1)), axis=1)))
    h = matrix(np.concatenate((np.zeros(D.shape[0]*D.shape[1]),  np.concatenate((mu,nu))+eps, -1*np.concatenate((mu,nu))+eps)))
    solvers.options['show_progress'] = False
    start = time.time()
    sol = solvers.lp(c,G,h)
    t_lp = time.time()-start

    X_lp = sol['x']
    X_lp = np.array(X_lp).reshape(ns,nt)

    return X_lp, t_lp

def save_config(path, n_source, n_target, eps_lp, eps1_dagp, eps2_dagp, step_dagp, rho, alpha, n_sims): 
    readme = 'The configuration of this simulation: \n' + \
    'n_source:                               ' + str(n_source) + '\n' + \
    'n_target:                               ' + str(n_target) + '\n' + \
    'eps_lp:                                 ' + str(eps_lp)  + '\n' + \
    'eps1_dagp:                              ' + str(eps1_dagp) + '\n' + \
    'eps2_dagp:                              ' + str(eps2_dagp) + '\n' + \
    'step_dagp:                              ' + str(step_dagp) + '\n' + \
    'DAGP_rho:                               ' + str(rho) + '\n' + \
    'DAGP_alpha:                             ' + str(alpha) + '\n' + \
    'n_simulations:                          ' + str(n_sims)                                    
    with open(path, "w") as text_file:
        text_file.write(readme)



