import numpy as np

import utils
import interactions
import dicke


def lrising_mags(wz, J0, alpha, gs, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)

    return dicke.mag_longitudinal(beta, wz, ws, np.sqrt(N) * lams, gs, N)


def lrising_mags_debug(wz, J0, alpha, gs, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)

    mags, uks = dicke.mag_longitudinal_debug(beta, wz, ws, np.sqrt(N) * lams, gs, N)

    return mags, uks


def Ys_f(wz, J0, alpha, gs, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)

    return dicke.Ys_f(beta, wz, ws, lams, gs, N)


def Y_f(wz, J0, alpha, beta):
    assert alpha <= 1
    return dicke.Y_f(beta, wz, 1 / J0)


def chiks_f(Y, ws):
    Dks = 1 / ws
    return Y / (1 - 2 * Dks * Y)


def chi_ij_f(i, j, wz, J0, alpha, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)
    lams = np.sqrt(N) * lams

    Ys = dicke.Ys_f(beta, wz, ws, lams, np.zeros(N), N)
    assert np.all(np.isclose(Ys, Ys[0], rtol=1e-3))
    Y = Ys[0]
    chiks = chiks_f(Y, ws)

    return 1 / N * np.sum(lams[i] * (chiks - Y) * lams[j]) + Y * (i == j)


def chi_ij_f_alt(i, j, wz, J0, alpha, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)
    lams = np.sqrt(N) * lams

    Y = Y_f(wz, J0, alpha, beta)
    chiks = chiks_f(Y, ws)

    return 1 / N * np.sum(lams[i] * (chiks - Y) * lams[j]) + Y * (i == j)


def chirs_f_alt(wz, J0, alpha, beta, N, M):
    J = interactions.powerlaw_pbc(N, alpha)
    J = interactions.shift(J, 0.0)
    J = J0 * interactions.rescale(J)

    ws, lams = utils.dicke_from_ising(J, 0.0)
    ws, lams = utils.truncate_dicke(ws, lams, M)
    lams = np.sqrt(N) * lams

    Y = Y_f(wz, J0, alpha, beta)
    chiks = chiks_f(Y, ws)

    chirs = 1 / N * np.sum(lams[0] * (chiks - Y) * lams, axis=1)
    chirs[0] += Y

    return chirs
