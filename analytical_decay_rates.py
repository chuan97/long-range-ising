import numpy as np
from scipy.optimize import curve_fit

def correlation_length_fit(r, l, A):
    return A * r**(-l) 

def correlation_decay_fit(x, m, n):
    return x*m + n

import algo

N = 100
fit_start = N//4
Ts = np.linspace(0.005, 1.4, 100)
betas = 1 / Ts
wz = 1
J0s = np.linspace(0.1, 0.65, 100)
alphas = np.linspace(0.01, 0.5, 15)
i = 0

rates_of_decay_rates = np.zeros((len(betas), len(J0s)))
for m, beta in enumerate(betas):
    print(m)
    for n, J0 in enumerate(J0s):
        crit_exp = []
        for alpha in alphas:
            M = round(N**np.tanh(2*alphas[-1]**(1/2)))
            #M = np.sqrt(N) * np.log(N)
            
            susc = algo.chirs_f_alt(wz, J0, alpha, beta, N, M)
            
            popt, pcov = curve_fit(correlation_length_fit, np.arange(fit_start, N//2, 1), susc[fit_start:N//2], p0=[1, 1])
            crit_exp.append(popt[0])

        popt, pcov = curve_fit(correlation_decay_fit, alphas[:2*len(alphas)//4], crit_exp[:2*len(alphas)//4], p0=(1, 0))
        rates_of_decay_rates[m, n] = popt[0]

np.savez(f'data/analytical_rates_of_decay_rates_{N}_{M}_{len(alphas)}_{len(Ts)}_{len(J0s)}_fixed_large', Ts=Ts, J0s=J0s, rates=rates_of_decay_rates)