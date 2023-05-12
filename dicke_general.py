import numpy as np
from scipy.optimize import minimize

def uks_self_consistent(uks, beta, ws, λs, f_m, args=None):
    return beta*np.sum(ws * uks**2) - f_m(uks, *args)

def uks_f(beta, ws, λs, f_m, args=None):
    return minimize(uks_self_consistent, x0=np.array([0.1]*len(ws)), args=(beta, ws, λs, f_m, args)).x

def u_self_consistent(u, beta, w0, f_m, args=None):
    return beta*w0 * u**2 - f_m(u, *args)

def u_f(beta, w0, f_m, args=None):
    return minimize(u_self_consistent, x0=np.array(0.1), args=(beta, w0, f_m, args)).x