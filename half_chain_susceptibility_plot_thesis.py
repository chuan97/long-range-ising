import matplotlib.pyplot as plt
import numpy as np

import algo
import plot

plot.set_rcParams(size=(10, 4), lw=2, fs=20)

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

N = 100

beta = 10
wz = 1
J0s = np.linspace(0.05, 0.5, 500)
alphas = [0.1, 0.5, 0.9]
gs = np.zeros(N)

i = 0
j = N // 2

cmap = plt.get_cmap("viridis")
colors = iter(cmap(np.linspace(0.9, 0.1, len(alphas))))
for alpha in alphas:
    M = round(N ** np.tanh(2 * alpha ** (1 / 2)))
    # M = N

    susc_exact_alt = np.array(
        [algo.chi_ij_f_alt(i, j, wz, J0, alpha, beta, N, M) for J0 in J0s]
    )

    c = next(colors)
    ax.plot(4 * J0s, N * susc_exact_alt, c=c, label=alpha)

ax.axvline(1.0, c="k", lw=0.5)
ax.set_yscale("log")
# ax.set_ylim(1e-1, 1e4)
# ax.set_xscale('log')
ax.set_ylabel(r"$N \chi_{N/2} \omega_z $")
ax.set_xlabel(r"$\Gamma / \omega_z$")
ax.legend(frameon=False, title=r"$\alpha$")

fig.savefig(
    f"plots/half_chain_susceptibility_{N}_{beta}_thesis.pdf",
    bbox_inches="tight",
    dpi=300,
)
