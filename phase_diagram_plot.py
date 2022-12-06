import matplotlib.pyplot as plt
import numpy as np

import plot

def arcth(x):
    return np.log((x + 1) / (x - 1)) / 2

plot.set_rcParams(size = (10, 8), lw = 2, fs = 20)
fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

data_rates = np.load('data/rates_of_decay_rates_10_5_5.npz')
Ts = data_rates['Ts']
J0s = data_rates['J0s']
rates = data_rates['rates']
cm = ax.pcolormesh(J0s, Ts, rates, vmin=0, vmax=1., cmap='viridis')   
fig.colorbar(cm, label=r'fit of $a$ ($\alpha_\chi = a \alpha + b ; \quad \chi_{0j} = A \cdot j^{-\alpha_\chi})$')

J0s_aux = np.linspace(0.25, 0.5, 100)
ax.set_xlim(0.05, 0.5)
ax.set_ylim(0.1, 1.5)
ax.plot(J0s_aux, 1 / (2 * arcth(4*J0s_aux)), c='w')
ax.set_xlabel(r'$\Gamma$')
ax.set_ylabel(r'$1/\beta$')
fig.savefig('plots/phase_diagram.pdf', dpi=300, bbox_inches='tight')
