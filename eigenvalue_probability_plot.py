import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from matplotlib.ticker import MultipleLocator

import plot
import interactions
import utils

plot.set_rcParams(size=(10, 9), lw=2, fs=20)

Ns = [100, 500, 1000]
alphas = [0.1, 0.8]

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.1, 0.9, len(Ns)))

fig, axes = plt.subplots(2, 2, constrained_layout=True)

for i, alpha in enumerate(alphas):
    ax = axes[0][i]
    N = 100

    Jbase = interactions.powerlaw_pbc(N, alpha)
    Jbase = interactions.shift(Jbase, 0.0)
    Jbase = interactions.rescale(Jbase)

    vals = eigh(Jbase, eigvals_only=True)
    # if j == 0:
    ks = np.arange(0, N)
    ax.plot(ks / N, vals[::-1], c=colors[0], lw=0, marker="o")
    Dks = [utils.Dk_exact(Jbase[: N // 2, 0], 2 * np.pi * k / N, N) for k in ks]
    Dks.sort(reverse=True)
    ax.plot(ks / N, Dks)
    ax.axhline(0, lw=0.5, c="k")

    if i == 0:
        # ax.legend(frameon=False, title=r'$N$')
        ax.set_ylabel("Eigenvalue " + r"$(D_k / \Gamma)$")
    if i == 1:
        ax.set_yticklabels([])

    ax.set_title(r"$\alpha=$ " + str(alpha))
    ax.set_xlabel(r"$k/N$")
    ax.axvline(N ** np.tanh(alpha ** (1 / 2)) / N, c="r")
    print(N, alpha, N ** np.tanh(3 * alpha), N ** np.tanh(3 * alpha) / N)

    ax = axes[1][i]

    data = []
    for N, color in zip(Ns, colors):
        Jbase = interactions.powerlaw_pbc(N, alpha)
        Jbase = interactions.shift(Jbase, 0.0)
        Jbase = interactions.rescale(Jbase)

        vals = eigh(Jbase, eigvals_only=True)
        data.append(vals)

    ax.hist(data, density=True, label=Ns, color=colors)

    ax.set_ylim(0.8e-2, 15)
    ax.set_yscale("log")
    if i == 0:
        ax.legend(frameon=False, title=r"$N$", loc="upper right")
        ax.set_ylabel("Probability density")
    if i == 1:
        ax.set_yticklabels([])
    ax.set_xlabel("Eigenvalue " + r"$(D_k / \Gamma)$")
    # ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fig.savefig("plots/eigenvalue_probability.pdf", bbox_inches="tight", dpi=300)
# fig.show()
