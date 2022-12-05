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