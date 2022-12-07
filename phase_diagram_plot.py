import matplotlib.pyplot as plt
import numpy as np

import plot

def arcth(x):
    return np.log((x + 1) / (x - 1)) / 2

plot.set_rcParams(size = (10, 8), lw = 2, fs = 20)
fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax = axes

data_rates = np.load('data/rates_of_decay_rates_20_100_100.npz')
Ts = data_rates['Ts']
J0s = data_rates['J0s']
rates = data_rates['rates']
cm = ax.pcolormesh(J0s, Ts, rates, vmin=0, vmax=1., cmap='viridis')   
label = r'fit of $a$ ($\alpha_\chi = a \alpha + b ; \quad \chi_{0j} = A \cdot j^{-\alpha_\chi})$'
cbar = fig.colorbar(cm, label=label, pad=0.01, aspect=40)
cbar.ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', r'$\geq 1.0$'])

J0s_aux = np.linspace(0.25, 0.65, 1000)
cmap = plt.get_cmap('viridis')
c1, c2, c3 = cmap(np.linspace(0.1, 0.9, 3))
ax.plot(J0s_aux, 1 / (2 * arcth(4*J0s_aux)), c='w', label=r'mean field $\alpha = 0$')

alpha = 0.8
N = 70
M = int(np.sqrt(N) * np.log(N))
wz = 1
#f_name = f'data/critical-J0-of-beta_{N}_{M}_{wz}_{alpha}.npz'
f_name = 'data/new_critical-J0-of-beta_70_35_1_0.2.npz'
data_line = np.load(f_name)
critical_J0 = data_line['critical_J0']
betas = data_line['betas']
print(len(betas))

data_x = [2, 2.49940119760479, 2.998802395209581, 3.498203592814371, 4.996407185628743]
data_y = [-0.0016194331983805377, 0.451821862348178, 0.6186234817813765, 0.7659919028340081, 1.1789473684210525]


plt.plot(critical_J0, 1/betas, c='r', label=r'analytical line $0 \leq \alpha \leq 1$')
marker_kwargs = {'marker': 'o',
                 'fillstyle': 'none',
                 'markevery': 1,
                 'markersize': 10,
                 'markeredgewidth': 1.5,
                }
plt.plot(np.array(data_x) / 8, data_y, c='red', lw=0, **marker_kwargs, label=r'numerical data $\alpha = 0.05$')

ax.set_xlim(0.1, 0.65)
ax.set_ylim(0.0, 1.4)
ax.set_xlabel(r'$\Gamma$')
ax.set_ylabel(r'$1/\beta$')
plt.legend(frameon=False)
fig.savefig('plots/phase_diagram.jpeg', dpi=300, bbox_inches='tight')
