import matplotlib.pyplot as plt
import numpy as np

import plot
from exact import (
    antiferro_hom_unfrustrated_big_spins,
    antiferro_hom_unfrustrated_small_spins,
    lanczos_ed,
)

plot.set_rcParams(size=(6.5, 8), lw=1.5, fs=14)

fig, axes = plt.subplots(2, 1, constrained_layout=True)

m_s = 7
m_b = 3

ax = axes[0]

N = 10
G = 1
s = 1 / 2
wx = 0.2 * s * G
wzs = np.linspace(0.001, 4, 50) * s * G

energies_small = []
energies_big = []
for wz in wzs:
    H_small = antiferro_hom_unfrustrated_small_spins(wx, wz, G, N, s)
    vals = lanczos_ed(H_small, k=m_s * 2)
    energies_small.append(vals / N)

    H_big = antiferro_hom_unfrustrated_big_spins(wx, wz, G, N, s)
    vals = lanczos_ed(H_big, k=m_b * 2)
    energies_big.append(vals / N)

energies_small = np.array(energies_small)[:, :-m_s]
energies_big = np.array(energies_big)[:, :-m_b]

ax.plot(wzs / (s * G), energies_small, c="k")
ax.plot(
    wzs / (s * G), energies_small[:, -1], c="k", label=r"$N$ $s=1/2$ spins"
)  # replot for single label
ax.plot(wzs / (s * G), energies_big, c="k", lw=0, marker="o", markevery=2)
ax.plot(
    wzs / (s * G),
    energies_big[:, -1],
    c="k",
    lw=0,
    marker="o",
    markevery=2,
    label=r"two $j=N/4$ spins",
)  # replot for single label

# ax.axvline(2, c='k', ls='dotted', zorder=0, lw=1)
# ax.axhline(-G*s**2, c='k', ls='dotted', zorder=0)
ax.set_xlabel(r"$\omega_z/(s\Gamma)$")
ax.set_ylabel(r"$E/(\Gamma N)$")
ax.text(
    0.98,
    0.95,
    rf"$\omega_x/(s\Gamma) = 0.2$",
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax.transAxes,
    fontsize=12,
)
ax.text(
    0.053,
    0.27,
    "(a)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
)
ax.grid(ls="--", alpha=0.75)
ax.legend(loc="lower left", fontsize=12)

ax = axes[1]

N = 10
G = 1
s = 1 / 2
wz = 0.2 * s * G
wxs = np.linspace(0.001, 2, 50) * s * G

energies_small = []
energies_big = []
for wx in wxs:
    H_small = antiferro_hom_unfrustrated_small_spins(wx, wz, G, N, s)
    vals = lanczos_ed(H_small, k=m_s * 2)
    energies_small.append(vals / N)

    H_big = antiferro_hom_unfrustrated_big_spins(wx, wz, G, N, s)
    vals = lanczos_ed(H_big, k=m_b * 2)
    energies_big.append(vals / N)

energies_small = np.array(energies_small)[:, :-m_s]
energies_big = np.array(energies_big)[:, :-m_b]

print(energies_small.shape)
print(energies_big.shape)

ax.plot(wxs / (s * G), energies_small, c="k")
ax.plot(
    wxs / (s * G), energies_small[:, -1], c="k", label="N s=1/2 spins"
)  # replot for single label
ax.plot(wxs / (s * G), energies_big, c="k", lw=0, marker="o", markevery=2)
ax.plot(
    wxs / (s * G),
    energies_big[:, -1],
    c="k",
    lw=0,
    marker="o",
    markevery=2,
    label="two J = N/4 spins",
)  # replot for single label

# ax.axvline(1, c='k', ls='dotted', zorder=0, lw=1)
# ax.axhline(-G*s**2, c='k', ls='dotted', zorder=0)
ax.set_xlabel(r"$\omega_x/(s\Gamma)$")
ax.set_yticks(np.arange(-0.2, -0.55, -0.1))
ax.set_ylabel(r"$E/(\Gamma N)$")
ax.text(
    0.98,
    0.95,
    rf"$\omega_z/(s\Gamma) = 0.2$",
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax.transAxes,
    fontsize=12,
)
ax.text(
    0.053,
    0.06,
    "(b)",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
    transform=ax.transAxes,
)
ax.grid(ls="--", alpha=0.75)
# ax.legend()


fig.savefig(f"plots/exact_antiferro.pdf", bbox_inches="tight", dpi=300)
