import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from matplotlib.ticker import MultipleLocator

import plot
import interactions
import utils

plot.set_rcParams(size = (10, 5), lw = 2, fs = 20)

Ns = np.arange(10, 100, 1)
alphas = [0.1, 0.5, 1.0, 1.5, 2.0]

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0.1, 0.9, len(Ns)))

fig, ax = plt.subplots(1, 1, constrained_layout=True)

for i, alpha in enumerate(alphas):
    avg_eig = []
    for N in Ns:
        Jbase = interactions.powerlaw_pbc(N, alpha)
        Jbase = interactions.shift(Jbase, 0.)
        Jbase = interactions.rescale(Jbase)

        vals = eigh(Jbase, eigvals_only=True)
        avg_eig.append(np.sum(vals) / N)
    avg_eig = np.array(avg_eig)    
    ax.plot(np.log10(Ns), 1 / avg_eig, label=alpha)
    
    #ax.set_xscale('log')
    #ax.set_yscale('log')
ax.set_xlim(1, 2)
ax.set_ylim(0, 20)
ax.legend()
    
fig.savefig('plots/eigenvalue_average.pdf', bbox_inches='tight', dpi=300)
        