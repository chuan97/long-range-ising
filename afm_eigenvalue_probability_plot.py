import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from matplotlib.ticker import MultipleLocator

import plot
import interactions
import utils

plot.set_rcParams(size=(10, 9), lw=2, fs=20)

Ns = [100, 500, 1000]
alphas = [0.2, 1.8]

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.1, 0.9, len(Ns)))

fig, axes = plt.subplots(2, 2, constrained_layout=True)

for i, alpha in enumerate(alphas):
    ax = axes[0][i]
    N = 100

    afmJ = interactions.powerlaw_pbc_afm(N, alpha)
    afmJ = interactions.shift(afmJ, 0.0)
    afmJ = interactions.rescale(afmJ)

    vals = eigh(afmJ, eigvals_only=True)
    # if j == 0:
    ks = np.arange(0, N)
    ax.plot(ks / N, vals[::-1], c=colors[0], lw=0, marker="o")
    Dks = [utils.Dk_exact(afmJ[: N // 2, 0], 2 * np.pi * k / N, N) for k in ks]
    Dks.sort(reverse=True)
    ax.plot(ks / N, Dks)
    ax.axhline(0, lw=0.5, c="k")

    if i == 0:
        # ax.legend(frameon=False, title=r'$N$')
        ax.set_ylabel("Eigenvalue " + r"$(D_k / \max \{D_k\})$")
    if i == 1:
        ax.set_yticklabels([])

    ax.set_title(r"$\alpha=$ " + str(alpha))
    ax.set_xlabel(r"$k/N$")
    ax.axvline(np.sqrt(N) * np.log(N) / N, c="r")

    ax = axes[1][i]

    data = []
    for N in Ns:
        Jbase = interactions.powerlaw_pbc(N, alpha)
        afmJ = np.zeros(Jbase.shape)
        for k in range(Jbase.shape[0]):
            afmJ += np.diag((-1) ** k * np.diag(Jbase, k=k), k=k)
            afmJ += np.diag((-1) ** k * np.diag(Jbase, k=-k), k=-k)

        afmJ = interactions.shift(afmJ, 0.0)
        # afmJ = interactions.rescale(afmJ)

        vals = eigh(afmJ, eigvals_only=True)
        data.append(vals / np.amax(vals))

    ax.hist(data, density=True, label=Ns, color=colors)

    ax.set_ylim(0.8e-2, 15)
    ax.set_yscale("log")
    if i == 0:
        ax.legend(frameon=False, title=r"$N$", loc="upper right")
        ax.set_ylabel("Probability density")
    if i == 1:
        ax.set_yticklabels([])
    ax.set_xlabel("Eigenvalue " + r"$(D_k / \max \{D_k\})$")
    # ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fig.savefig("plots/afm_eigenvalue_probability.pdf", bbox_inches="tight", dpi=300)
# fig.show()
