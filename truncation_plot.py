import matplotlib.pyplot as plt
import numpy as np

import plot
import interactions
import utils

plot.set_rcParams(size=(10, 5), lw=2, fs=20)
cmap = plt.get_cmap("viridis")

N = 1000
Ms = [N, int(np.sqrt(N) * np.log(N))]

ref_idx = N // 2

alphas = [0.2, 1.8]

fig, axes = plt.subplots(1, 2, constrained_layout=True)

for i, alpha in enumerate(alphas):
    ax = axes[i]

    if i == 1:
        # inset axes....
        axins = ax.inset_axes([0.3, 0.3, 0.67, 0.67])
        # sub region of the original image
        x1, x2, y1, y2 = 0, 45, -0.03, 0.15
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])

    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 1e-6)
    J = interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)

    print(len(ws))
    colors = iter(cmap(np.linspace(0.25, 0.6, len(Ms))))
    labels = iter(["N", "log(N) sqrt(N)"])
    for M in Ms:
        ws_truncated, lams_truncated = utils.truncate_dicke(ws, lams, M)
        # ws_truncated = np.array([w for i, w in enumerate(ws) if len(ws) - i <= M])
        # print(len(ws_truncated))
        # lams_truncated = np.array([vect for i, vect in enumerate(lams.T) if len(ws) - i <= M]).T
        c = next(colors)
        J_truncated = utils.ising_from_dicke(ws_truncated, lams_truncated.T)
        ax.plot(
            np.arange(1, N - ref_idx),
            J_truncated[ref_idx, ref_idx + 1 :] / J[ref_idx, ref_idx + 1],
            c=c,
            label="M=" + next(labels),
        )

        if i == 1:
            axins.plot(
                np.arange(1, N - ref_idx), J_truncated[ref_idx, ref_idx + 1 :], c=c
            )

    # ax.plot(np.arange(1, N - ref_idx), J[ref_idx, ref_idx+1:], c='k', ls='dashed', label=r'$J_{i \neq j} \propto |i - j|^{-\alpha}$')

    if i == 0:
        ax.legend(frameon=False)
        ax.set_ylabel(r"$J_M(r) / J(1)$")

        ax.set_xlim(0, 400)
    ax.set_xlabel(r"$r$")
    # ax.set_ylim(0.0, 1.0)
    if i == 1:
        ax.indicate_inset_zoom(axins, edgecolor="black")
        # ax.set_yticklabels([])
        ax.set_xlim(0, 400)
    # plt.loglog()
    # plt.xscale('log')
    ax.set_title(r"$\alpha=$ " + str(alpha))
    handles, labels = ax.get_legend_handles_labels()


# fig.legend(handles[:-1], labels[:-1], loc='upper right', bbox_to_anchor=(0.95, 0.8)) #, title='Aproximations'
# plt.tight_layout()
plt.savefig("plots/truncation_of_modes.pdf", bbox_inches="tight", dpi=300)
