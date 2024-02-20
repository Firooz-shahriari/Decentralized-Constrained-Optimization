import numpy as np

class OT():
    """
    Optimal Tranport Class
        D:  cost matrix
        N:  number of nodes
        mu: source distribution
        nu: target distribution
    """
    def __init__(self, D, N, mu, nu, reg_constant):
        self.D = D
        self.m = D.shape[0]
        self.n = D.shape[1]
        self.N = N
        self.X = np.zeros((self.N, self.m, self.n))
        self.mu = mu
        self.nu = nu
        self.X_train = None
        self.Y_train = None
        self.row_ind, self.col_ind, self.nr = self.distribute(self.m, self.n, self.N)
        self.cons = reg_constant

    ## distribute the functions and constraints as uniform as possible
    def distribute(self, m, n, M):   #m:row // n:col // M:nodes
        l = np.ceil((m+n)/M)
        m_prime = np.ceil(m/l)
        n_prime = np.ceil(n/l)
        l_prime = l + np.abs( m_prime+n_prime-M )
        a_1 = np.ceil(m/l_prime)   # number of agents needed for rows
        a_2 = np.ceil(n/l_prime)   # number of agents needed for cols
        l_1 = np.ceil(m/a_1)       # max rows in each node
        l_2 = np.ceil(n/a_2)       # max cols in each node
        row_ind = []
        col_ind = [] 
        for i in range(M):
            row_ind.append([j+l_1*i       for j in range(np.int(l_1)) if i< a_1 and j+l_1*i<m       ])
            col_ind.append([j+l_2*(i-a_1) for j in range(np.int(l_2)) if i>=a_1 and j+l_2*(i-a_1)<n ])
        return row_ind, col_ind, a_1

    def F_val(self, theta):
        return np.sum(np.multiply(self.D,theta)) + self.cons*np.sum(np.multiply(theta,theta))

    def localgrad(self, agent_num, theta):
        loc_grad = np.zeros((self.m, self.n))

        rows = np.asarray(self.row_ind[agent_num]).astype(int)
        cols = np.asarray(self.col_ind[agent_num]).astype(int)

        if agent_num < self.nr:
            loc_grad[rows] = self.D[rows]/2 + (self.cons)*theta[rows]
        else:
            loc_grad[:,cols] = self.D[:,cols]/2 + (self.cons)*theta[:,cols]

        return loc_grad  

    def networkgrad(self, theta):
        netgrad = np.zeros([self.N, self.m, self.n])
        for i in range(self.N):
            netgrad[i,:,:] = self.localgrad(i, theta[i])
        return netgrad

    def local_projection(self, theta, agent_num):
        loc_proj = theta[agent_num,:,:]

        rows = np.asarray(self.row_ind[agent_num]).astype(int)
        cols = np.asarray(self.col_ind[agent_num]).astype(int)

        if agent_num < self.nr:
            for i in rows: 
                loc_proj[i,:] = simplex_projection(loc_proj[i,:], self.mu[i])
        else:
            for i in cols:
                loc_proj[:,i] = simplex_projection(loc_proj[:,i], self.nu[i])

        return loc_proj
        
    def network_projection(self, theta):
        net_proj = theta
        for i in range(self.N):
            net_proj[i,:,:] = self.local_projection(theta, i)
        return net_proj

## Matrix Frobenius Inner Product
def MIP(A,B):
    return np.trace(np.matmul(A,np.transpose(B)))

# Projection algorithms, Condat[2016], sorting Alg
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

