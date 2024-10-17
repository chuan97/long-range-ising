import matplotlib.pyplot as plt
import numpy as np

import plot


def arcoth(x):
    return np.log((x + 1) / (x - 1)) / 2


plot.set_rcParams(size=(10, 8.5), lw=3, fs=20)
fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes
# plt.axhline(0.02, c='k', lw=0.75)

# data_rates = np.load('data/latest_rates_of_decay_rates_120_52_15_100_100.npz')
data_rates = np.load(
    "data/analytical_rates_of_decay_rates_100_60_15_100_100_fixed_large.npz"
)
alt_files = [
    "data/latest_rates_of_decay_rates_110_49_15_100_100.npz",
]
alt_files = []
Ts = data_rates["Ts"]
J0s = data_rates["J0s"]
rates = data_rates["rates"]

# correct corrupted data points
for fname in alt_files:
    alt_data = np.load(fname)
    alt_rates = alt_data["rates"]

    for idx, val in np.ndenumerate(rates):
        if val < 0 or val > 1:
            rates[idx] = alt_rates[idx]

print(np.amin(rates), np.amax(rates))
rates[rates > 1] = 1
rates[rates < 0] = 0
cm = ax.pcolormesh(4 * J0s, Ts, rates, cmap="viridis", vmin=0, vmax=1)
# label = r'fit of $a$ ($\alpha_\chi = a \alpha + b ; \quad \chi_{0j} = A \cdot j^{-\alpha_\chi})$'
label = r"$a$ (fitted from $\alpha_\chi = a \alpha + b)$"
cbar = fig.colorbar(cm, pad=0.02, aspect=40)
# cbar.ax.set_yticks(np.linspace(0.0, 1.0, 6))
# cbar.ax.set_yticklabels(np.arange(0.0, 1.1, 0.2))

c = plt.get_cmap("magma")(np.linspace(0.1, 0.9, 5))[3]

J0s_aux = np.linspace(0.25, 0.65, 1000)
cmap = plt.get_cmap("viridis")
c1, c2, c3 = cmap(np.linspace(0.1, 0.9, 3))
ax.plot(
    4 * J0s_aux,
    1 / (2 * arcoth(4 * J0s_aux)),
    c="r",
    label=r"analytical line $\alpha < 1$",
)

# f_name = 'data/alt_new_critical-J0-of-beta_70_35_1_0.5_52_30.npz'
# data_line = np.load(f_name)
# critical_J0 = data_line['critical_J0']
# betas = data_line['betas']
# print(len(betas))
# plt.plot(critical_J0, 1/betas, c='r', ls='--', label=r'analytical line $0 \leq \alpha \leq 1$')

marker_kwargs = {
    "marker": "o",
    "fillstyle": "none",
    "markevery": 1,
    "markersize": 12,
    "markeredgewidth": 2,
}
data_x = [2, 2.49940119760479, 2.998802395209581, 3.498203592814371, 4.996407185628743]
data_y = [
    -0.0016194331983805377,
    0.451821862348178,
    0.6186234817813765,
    0.7659919028340081,
    1.1789473684210525,
]

plt.plot(
    4 * np.array(data_x) / 8,
    data_y,
    c="r",
    lw=0,
    **marker_kwargs,
    label=r"numerical data $\alpha = 0.05$"
)

ax.set_xlim(0.1 * 4, 0.65 * 4)
ax.set_xticks(np.arange(0.5, 3, 0.5))
ax.set_ylim(0.0, 1.4)
ax.set_xlabel(r"$\Gamma / \omega_z$")
ax.set_ylabel(r"$1/(\beta \omega_z)$")
ax.set_title(label, pad=15)
ax.legend(
    frameon=False,
    framealpha=0.4,
    facecolor="Gray",
    edgecolor="Gray",
    labelcolor="w",
    loc="upper center",
)
fig.savefig("plots/phase_diagram_thesis.jpeg", dpi=300, bbox_inches="tight")
