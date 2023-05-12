import numpy as np

def eps(uks, wz, lams, hs):
    aux = 2*np.sum(lams*uks, axis=1) + hs
    return 0.5 * np.sqrt(wz**2 + 4*aux**2)

def eps_homo(u, wz):
    return 0.5 * np.sqrt(wz**2 + 16*u**2)

def f_ising(uks, beta, wz, lams, hs, N):
    return 1/N * np.sum(np.log(2 * np.cosh(beta * eps(uks, wz, lams, hs))))

def f_ising_homo(u, beta, wz):
    return np.log(2 * np.cosh(beta * eps_homo(u, wz)))

def f_ising_antihomo(u, beta, wz):
    return np.log(2 * np.cosh(beta * eps_homo(u, wz)))