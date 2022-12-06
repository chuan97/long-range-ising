import matplotlib.pyplot as plt
import numpy as np

import plot
import interactions


plot.set_rcParams(size = (7, 7), lw = 2, fs = 20)

cmap = plt.get_cmap('viridis')

alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
Ns = np.arange(10, 100, 1)

colors = cmap(np.linspace(0, 1, 10))
for i, alpha in enumerate(alphas):
    Ss = np.array([np.sum(interactions.powerlaw_pbc(N, alpha)[0]) for N in Ns])
    
    c = colors[5] if alpha == 1.0 else 'k'
    plt.plot(np.log10(Ns), Ss, label=alpha, c=c)
    
    label = str(alpha)
    xpos = np.log10(Ns)[10 * (i + 1)]
    if i == 0:
        label = r'$\alpha = $' + label
        xpos = np.log10(Ns)[10 * (i + 1) - 4]
    plt.text(xpos, Ss[10 * (i + 1)], label, fontsize=15, c=c, backgroundcolor='white')

#plt.plot(Ns, Ns, c='k', ls='dashed')
#plt.arrow(11, 13, 5, -11, width = 0.1, head_width=0.5, facecolor='k', zorder=6)
#plt.annotate(r'$\alpha$', xy=(np.log10(17), 1), xytext=(np.log10(11), 13), arrowprops=dict(arrowstyle="->", linestyle='--', linewidth=1.0))
#plt.text(17, 1, r'$\alpha$')
#plt.xscale('log')
plt.xlim(1, 2)
plt.ylim(0, 20)
plt.xlabel(r'$\log_{10}(N)$')
plt.ylabel(r'$\sum_j \, \, \, \tilde{J}_{ij}$')
#plt.legend()
plt.tight_layout()
plt.savefig('plots/S.pdf', bbox_inches='tight', dpi=300)