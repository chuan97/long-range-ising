import matplotlib.pyplot as plt
import numpy as np

import plot
import algo


plot.set_rcParams(size = (10, 4), lw = 2, fs = 20)

fig, axes = plt.subplots(1, 1, constrained_layout=True)


ax = axes

N = 70

beta = 10
wz = 1
J0s = np.linspace(0.05, 0.5, 100)
alphas = [0.1, 0.5, 0.9]

dg = 1e-3 * wz

i = 0
j = N//2

cmap = plt.get_cmap('viridis')
colors = iter(cmap(np.linspace(0.9, 0.1, len(alphas))))
for alpha in alphas:
    M = round(N**np.tanh(2*alpha**(1/2)))
    susc = []
    susc_exact = []

    for J0 in J0s:
        gs = np.zeros(N)
        mxs0 = algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)
        gs[i] = dg
        mxs1 = algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)
        susc.append(((mxs1 - mxs0) / dg)[j])
        susc_exact.append(algo.chi_ij_f(i, j, wz, J0, alpha, beta, N, M))
    
    c = next(colors)
    ax.plot(J0s, susc, c=c, label=alpha)
    ax.plot(J0s, susc_exact, c=c, lw=0, marker='o', markevery=2)

ax.axvline(0.25, c='k', lw=0.5)
ax.set_yscale('log')
#ax.set_ylim(1e-4, 1e2)
#ax.set_xscale('log')
ax.set_ylabel(r'$\chi_{N/2} \omega_z $')
ax.set_xlabel(r'$\Gamma / \omega_z$')
ax.legend(frameon=False, title=r'$\alpha$')

if j == N//2:
    fig.savefig(f'plots/comparison_half_chain_susceptibility_{N}_{beta}.pdf', bbox_inches='tight', dpi=300)
elif j == 1:
    fig.savefig(f'plots/comparison_first_neighbour_susceptibility_{N}_{beta}.pdf', bbox_inches='tight', dpi=300)