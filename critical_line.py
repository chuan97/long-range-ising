import numpy as np

import utils
import algo

def ising_mag(J0s, wz, alpha, gs, beta, N, M):
    return np.abs(np.array([algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)[0] for J0 in J0s]))

N = 70
#M = int(np.sqrt(N) * np.log(N))
M = N
wz = 1
Ts = np.linspace(0.01, 1.4, 52)
betas = 1/Ts
J0s = np.linspace(0.1, 1.0, 30)
alpha = 1.5
gs = 0.0 * np.ones(N)

critical_J0s = []
for beta in betas:
    critical_J0 = utils.critical_point_recursive_refinement(ising_mag, J0s, [wz, alpha, gs, beta, N, M], rounds=3)
    critical_J0s.append(critical_J0)

np.savez(f'data/alt_new_critical-J0-of-beta_{N}_{M}_{wz}_{alpha}_{len(Ts)}_{len(J0s)}.npz', betas=betas, critical_J0=critical_J0s)
