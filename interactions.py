import numpy as np
from scipy.linalg import eigh

def powerlaw_pbc(N, alpha):
    J = np.zeros((N, N))
   
    for idx, _ in np.ndenumerate(J):
        i, j = idx
        J[idx] = 0 if i == j else float(min(np.abs(j - i), N - np.abs(j - i))) ** (-alpha)
            
    return J

def shift(J, epsilon):
    vals = eigh(J, eigvals_only=True)
    return J - np.eye(J.shape[0]) * (vals[0] - epsilon)

def rescale(J):
    S = np.sum(np.abs(J[0]))
    return J / S

def powerlaw_obc_2D(L, alpha):
    N = L**2
    J = np.zeros((N, N))
   
    for idx, _ in np.ndenumerate(J):
        i, j = idx
        r_i = r_from_idx_2D(i, L)
        r_j = r_from_idx_2D(j, L)
        r_ij = r_i - r_j
        mod_r_ij = np.sqrt(np.sum(r_ij**2))
        J[idx] = 0 if i == j else mod_r_ij ** (-alpha)
            
    return J

def powerlaw_pbc_2D(L, alpha):
    N = L**2
    J = np.zeros((N, N))
   
    for idx, _ in np.ndenumerate(J):
        i, j = idx
        r_i = r_from_idx_2D(i, L)
        r_j = r_from_idx_2D(j, L)
        r_ij = np.abs(r_i - r_j)
        r_ij = np.minimum(r_ij, L - r_ij)
        mod_r_ij = np.sqrt(np.sum(r_ij**2))
        J[idx] = 0 if i == j else mod_r_ij ** (-alpha)
            
    return J

def r_from_idx_2D(idx, L):
    return np.array(divmod(idx, L))