import numpy as np
from scipy.optimize import basinhopping, minimize

def kernel(xs, λs, gs):
    return 2*np.sum(λs*xs, axis=1) + gs

def mag_longitudinal(beta, wz, ws, λs, gs, N):
    def func(xs):
        aux = kernel(xs, λs, gs)
        return beta * np.sum(ws * xs**2) - 1/N * np.sum(np.log(2 * np.cosh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2))))
    niter = 5
    #xs = basinhopping(func, x0=np.array([0.5]*len(ws)), niter=niter).x
    xs = minimize(func, x0=np.array([0.1]*len(ws))).x
    #print(xs)
    aux = kernel(xs, λs, gs)
    #print(xs)
    return 0.5 * np.tanh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2)) * 4 * aux / np.sqrt(wz**2 + 4*aux**2)