import matplotlib.pyplot as plt
import numpy as np

import plot
import algo


plot.set_rcParams(size=(10, 8), lw=2, fs=20)

fig, axes = plt.subplots(2, 1, constrained_layout=True)


ax = axes[0]

N = 70
M = int(np.sqrt(N) * np.log(N))

beta = 10
wz = 1
J0s = np.linspace(0.05, 0.5, 100)
alpha = 0.5

gs = 0.0 * np.ones(N)

mags = np.array([algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)[0] for J0 in J0s])

marker_kwargs = {
    "c": "k",
    "lw": 0,
    "marker": "o",
    "fillstyle": "none",
    "markevery": 3,
    "markersize": 10,
    "markeredgewidth": 1.5,
}
ax.plot(J0s, np.abs(mags), **marker_kwargs, label=r"$\alpha =$ " + str(alpha))
ax.plot(
    J0s,
    np.nan_to_num(np.sqrt(1 - (wz / (4 * J0s)) ** 2)),
    c="k",
    label=r"$\alpha = 0$ (all-to-all)",
)
ax.axvline(0.25, c="k", lw=0.5)
# ax.set_xlabel(r'$\Gamma$')
ax.set_ylabel(r"$\langle \sigma^x \rangle$")
ax.legend(frameon=False)


ax = axes[1]

alphas = [0.1, 0.5, 0.9]

dg = 5e-3 * wz

i = 0
j = N // 2

cmap = plt.get_cmap("viridis")
colors = iter(cmap(np.linspace(0.9, 0.1, len(alphas))))
for alpha in alphas:
    susc = []

    for J0 in J0s:
        gs = 0.0 * np.ones(N)
        mxs0 = algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)
        gs[i] = dg
        mxs1 = algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)
        susc.append(((mxs1 - mxs0) / dg)[j])

    ax.plot(J0s, susc, c=next(colors), label=alpha)

ax.axvline(0.25, c="k", lw=0.5)
ax.set_yscale("log")
ax.set_ylabel(r"$\chi_{N/2}$")
ax.set_xlabel(r"$\Gamma$")
ax.legend(frameon=False, title=r"$\alpha$")


fig.savefig("plots/observables.pdf", bbox_inches="tight", dpi=300)
