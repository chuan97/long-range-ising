import numpy as np
from scipy.optimize import minimize

def kernel(xs, λs, gs):
    return 2*np.sum(λs*xs, axis=1) + gs

def uks_self_consistent(uks, beta, wz, ws, λs, gs, N):
    aux = kernel(uks, λs, gs)
    return beta*np.sum(ws * uks**2) - 1/N*np.sum(np.log(2 * np.cosh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2))))

def uks_f(beta, wz, ws, λs, gs, N):
    return minimize(uks_self_consistent, x0=np.array([0.1]*len(ws)), args=(beta, wz, ws, λs, gs, N)).x

def u_self_consistent(u, beta, wz, w0):
    return beta*w0*u**2 - np.log(2 * np.cosh(0.5 * beta * np.sqrt(wz**2 + 16*u**2)))

def u_f(beta, wz, w0):
    return minimize(u_f, x0=0.1, args=(beta, wz, w0)).x

def mag_longitudinal(beta, wz, ws, λs, gs, N):
    uks = uks_f(beta, wz, ws, λs, gs, N)
    aux = kernel(uks, λs, gs)
    #print(xs)
    return 0.5 * np.tanh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2)) * 4 * aux / np.sqrt(wz**2 + 4*aux**2)

def mag_longitudinal_old(beta, wz, ws, λs, gs, N):
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

def mag_longitudinal_hessian_debug(beta, wz, ws, λs, gs, N):
    def func(xs):
        aux = kernel(xs, λs, gs)
        return beta * np.sum(ws * xs**2) - 1/N * np.sum(np.log(2 * np.cosh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2))))
    niter = 5
    #xs = basinhopping(func, x0=np.array([0.5]*len(ws)), niter=niter).x
    xs = minimize(func, x0=np.array([0.1]*len(ws))).x
    print(xs)

    u = xs[-1]
    e = 0.5 * np.sqrt(wz**2 + 16*u**2)
    max_H_eigval = u**2/e**2 * (16*beta**2 - 4*ws[-1]**2*e**2*beta**2 - 8*ws[-1]*beta)
    assert max_H_eigval <= 0
    #print(xs)
    aux = kernel(xs, λs, gs)
    return 0.5 * np.tanh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2)) * 4 * aux / np.sqrt(wz**2 + 4*aux**2)

def mag_longitudinal_debug(beta, wz, ws, λs, gs, N):
    import matplotlib.pyplot as plt
    #print(λs)
    #print(np.sum(λs, axis=0) / N)
    #plt.imshow(λs)
    #plt.colorbar()
    #plt.show()
    def func(xs):
        aux = kernel(xs, λs, gs)
        return beta * np.sum(ws * xs**2) - 1/N * np.sum(np.log(2 * np.cosh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2))))
    niter = 5
    #xs = basinhopping(func, x0=np.array([0.5]*len(ws)), niter=niter).x
    xs = minimize(func, x0=np.array([0.1]*len(ws))).x
    #print(xs)
    aux = kernel(xs, λs, gs)
    #print(xs)
    return 0.5 * np.tanh(0.5 * beta * np.sqrt(wz**2 + 4*aux**2)) * 4 * aux / np.sqrt(wz**2 + 4*aux**2), xs

def Ys_f(beta, wz, ws, λs, gs, N):
    uks = uks_f(beta, wz, ws, λs, gs, N)
    #print(uks[-1])
    aux = kernel(uks, λs, gs)
    eps = 0.5 * np.sqrt(wz**2 + 4*aux**2)
    
    return (1 - np.tanh(beta * eps)**2)*beta*(aux / eps)**2 + np.tanh(beta * eps)*(1/eps - (aux**2 / eps**3)) 

def Y_f(beta, wz, w0):
    u = u_f(beta, wz, w0)
    aux = 2*u
    eps = 0.5 * np.sqrt(wz**2 + 4*aux**2)
    return (1 - np.tanh(beta * eps)**2)*beta*(aux / eps)**2 + np.tanh(beta * eps)*(1/eps - (aux**2 / eps**3)) 
    
