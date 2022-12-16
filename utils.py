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

def critical_point_recursive_refinement(f_order_parameter, critical_parameter, other_args, rounds=2, *, verbose=False):
    L = len(critical_parameter)
    op = f_order_parameter(critical_parameter, *other_args)
    rel_diff = (op[1:] - op[:-1]) #/ op[:-1]
    idx = np.argmax(rel_diff)
    cp = critical_parameter[idx]
    
    if rounds == 1:
        return cp
    
    else:
        if verbose:
            print(f'Rounds to go: {rounds - 1}, current critical point:  {cp} ...')

        new_range = min(idx, L - idx) // 2
        range_min = critical_parameter[idx - new_range]
        range_max = critical_parameter[idx + new_range]
        step = (cp - range_min) / (L // 2)
        
        if step == 0:
            return cp
        
        new_critical_parameter = np.arange(range_min, range_max, step)
        
        return critical_point_recursive_refinement(f_order_parameter, new_critical_parameter, other_args, rounds - 1, verbose=verbose)