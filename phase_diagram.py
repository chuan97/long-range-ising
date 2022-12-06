import numpy as np
from scipy.optimize import curve_fit

def correlation_length_fit(r, l, A):
    return A * r**(-l) 

def correlation_decay_fit(x, m, n):
    return x*m + n

import algo

N = 70
M = int(np.sqrt(N) * np.log(N))
Ts = np.linspace(0.05, 1.4, 100)
betas = 1 / Ts
wz = 1
J0s = np.linspace(0.1, 0.65, 100)
alphas = np.linspace(0.3, 0.99, 20)
gs = 0.0 * np.ones(N)
dg = 5e-3 * wz
i = 0

rates_of_decay_rates = np.zeros((len(betas), len(J0s)))
for m, beta in enumerate(betas):
    print(m)
    for n, J0 in enumerate(J0s):
        crit_exp = []
        for alpha in alphas:
            gs = 0.0 * np.ones(N)
            mxs0 = algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)
            gs[i] = dg
            mxs1 = algo.lrising_mags(wz, J0, alpha, gs, beta, N, M)
            susc = ((mxs1 - mxs0) / dg)

            fit_start = N//4
            popt, pcov = curve_fit(correlation_length_fit, np.arange(fit_start, N//2, 1), susc[fit_start:N//2], p0=[1, 1])
            crit_exp.append(popt[0])

        popt, pcov = curve_fit(correlation_decay_fit, alphas[:3*len(alphas)//4], crit_exp[:3*len(alphas)//4], p0=(1, 0))
        rates_of_decay_rates[m, n] = popt[0]

np.savez(f'data/rates_of_decay_rates_{len(alphas)}_{len(Ts)}_{len(J0s)}', Ts=Ts, J0s=J0s, rates=rates_of_decay_rates)