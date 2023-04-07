import numpy as np

import utils
import interactions
import dicke

def lrising_mags(wz, J0, alpha, gs, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)
    
    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)
    
    return dicke.mag_longitudinal(beta, wz, ws, np.sqrt(N) * lams, gs, N)

def lrising_mags_debug(wz, J0, alpha, gs, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)
    
    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)
    
    mags, uks = dicke.mag_longitudinal_debug(beta, wz, ws, np.sqrt(N) * lams, gs, N)
    
    return mags, uks