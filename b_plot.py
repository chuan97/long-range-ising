import matplotlib.pyplot as plt
import numpy as np

import plot
import interactions


plot.set_rcParams(size = (7, 3), lw = 2, fs = 20)

cmap = plt.get_cmap('viridis')

alphas = [0.1, 0.5, 0.9]
Ns = np.arange(10, 100, 1)

colors = iter(cmap(np.linspace(0.1, 0.9, 3)))
for alpha in alphas:
    bs = []
    for N in Ns:
        Jbase = interactions.powerlaw_pbc(N, alpha)
        _, b = interactions.shift(Jbase, 0., return_shift=True)
        bs.append(b)
    bs = np.array(bs)
    
    plt.plot(Ns, bs, label=alpha, c=next(colors))

#plt.plot(Ns, Ns, c='k', ls='dashed')
#plt.arrow(11, 13, 5, -11, width = 0.1, head_width=0.5, facecolor='k', zorder=6)
#plt.annotate(r'$\alpha$', xy=(np.log10(17), 1), xytext=(np.log10(11), 13), arrowprops=dict(arrowstyle="->", linestyle='--', linewidth=1.0))
#plt.text(17, 1, r'$\alpha$')
#plt.xscale('log')
# plt.xlim(1, 2)
# plt.ylim(0, 20)
plt.xlabel(r'$N$')
plt.ylabel(r'$b$')
plt.legend()
plt.tight_layout()
plt.savefig('plots/b.pdf', bbox_inches='tight', dpi=300)