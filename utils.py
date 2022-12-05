import numpy as np
from scipy.linalg import eigh

def ising_from_dicke(ws, lams):
    J = np.zeros((lams.shape[1], lams.shape[1])) 
    
    for w, lam in zip(ws, lams):
        J += np.kron(np.reshape(lam, (lams.shape[1], 1)), lam) / w
        
    return J

def dicke_from_ising(J, threshold = 1e-6):
    vals, vects = eigh(J)
    ws = np.array([1/val for val in vals if np.abs(val) >= threshold])
    lams = np.array([vect for vect, val in zip(vects.T, vals) if np.abs(val) >= threshold])
    
    return ws, lams.T

def truncate_dicke(ws, lams, M):
    ws_truncated = np.array([w for i, w in enumerate(ws) if len(ws) - i <= M])
    lams_truncated = np.array([vect for i, vect in enumerate(lams.T) if len(ws) - i <= M]).T
    
    return ws_truncated, lams_truncated