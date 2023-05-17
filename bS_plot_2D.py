import matplotlib.pyplot as plt
import numpy as np

import plot
import interactions

plot.set_rcParams(size = (10, 5.5), lw = 3, fs = 20)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
ax = axes[0]

cmap = plt.get_cmap('viridis')

alphas = [0.5, 1, 2.0, 3.0, 4.0]
Ns = np.arange(2, 20, 1)

colors = cmap(np.linspace(0, 1, 10))
for i, alpha in enumerate(alphas):
    print(alpha)
    Ss = np.array([np.sum(interactions.powerlaw_pbc_2D(N, alpha)[0]) for N in Ns])
    
    c = colors[5] if alpha == 2.0 else 'k'
    ax.plot(np.log10(Ns), Ss, label=alpha, c=c)
    
    # label = str(alpha)
    # xpos = np.log10(Ns)[10 * (i + 1)]
    # if i == 0:
    #     label = r'$\alpha = $' + label
    #     xpos = np.log10(Ns)[2 * (i + 1) - 4]
    # ax.text(xpos, Ss[10 * (i + 1)], label, fontsize=15, c=c, backgroundcolor='white')

#plt.plot(Ns, Ns, c='k', ls='dashed')
#plt.arrow(11, 13, 5, -11, width = 0.1, head_width=0.5, facecolor='k', zorder=6)
#plt.annotate(r'$\alpha$', xy=(np.log10(17), 1), xytext=(np.log10(11), 13), arrowprops=dict(arrowstyle="->", linestyle='--', linewidth=1.0))
#plt.text(17, 1, r'$\alpha$')
#plt.xscale('log')
ax.set_xlim(np.log10(Ns[0]), np.log10(Ns[-1]))
#ax.set_ylim(0, 20)
ax.set_xlabel(r'$\log_{10}(N)$')
#ax.set_ylabel(r'$\sum_j \, \, \, \tilde{J}_{ij}$')
ax.set_ylabel(r'$\tilde N$')

ax = axes[1]

alphas = [0.5, 1, 1.5]
Ns = np.arange(2, 20, 1)

colors = iter(cmap(np.linspace(0.1, 0.9, 3)))
for alpha in alphas:
    print(1/alpha)
    bs = []
    for N in Ns:
        Jbase = interactions.powerlaw_pbc_2D(N, alpha)
        _, b = interactions.shift(Jbase, 0., return_shift=True)
        bs.append(b)
    bs = np.array(bs)
    
    c = next(colors)
    ax.plot(Ns, bs, label=alpha, c=c)

#plt.plot(Ns, Ns, c='k', ls='dashed')
#plt.arrow(11, 13, 5, -11, width = 0.1, head_width=0.5, facecolor='k', zorder=6)
#plt.annotate(r'$\alpha$', xy=(np.log10(17), 1), xytext=(np.log10(11), 13), arrowprops=dict(arrowstyle="->", linestyle='--', linewidth=1.0))
#plt.text(17, 1, r'$\alpha$')
#plt.xscale('log')
# plt.xlim(1, 2)
# plt.ylim(0, 20)
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$b$')
ax.legend(title=r'$\alpha$', loc='lower right')
fig.savefig('plots/bS_2D.pdf', bbox_inches='tight', dpi=300)