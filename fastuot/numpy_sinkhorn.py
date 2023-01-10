import numpy as np
from scipy.special import logsumexp
from .uot1d import rescale_potentials


def sinkx(C, f, a, eps):
    return - eps * logsumexp(np.log(a)[:, None] + (f[:, None] - C) / eps,
                             axis=0)


def sinky(C, g, b, eps):
    return - eps * logsumexp(np.log(b)[None, :] + (g[None, :] - C) / eps,
                             axis=1)


def softmin(a, f, rho):
    return - rho * logsumexp(np.log(a) - f / rho)


def aprox(f, eps, rho):
    return (1. / (1. + (eps / rho))) * f


def dual_score_ent(f, g, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho

    def phi(x, s):
        return -s * (np.exp(-x / s) - 1)

    return np.sum(a * phi(f, rho)) + np.sum(b * phi(g, rho2)) + np.sum(
        a[:, None] * b[None, :] * phi(C - f[:, None] - g[None, :], eps))


def balanced_loop(f, a, b, C, eps):
    # Update on G
    g = sinkx(C, f, a, eps)
    # Update on F
    f = sinky(C, g, b, eps)
    return f, g


def f_sinkhorn_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)
    return f, g


def h_sinkhorn_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        g = aprox(sinkx(C, f, a, eps), eps, rho) \
            - 0.5 * (eps / (eps + rho)) * softmin(a, f, rho)
        g = g + (eps / (eps + 2 * rho)) * softmin(b, g, rho)
        f = aprox(sinky(C, g, b, eps), eps, rho) \
            - 0.5 * (eps / (eps + rho)) * softmin(b, g, rho)
        f = f + (eps / (eps + 2 * rho)) * softmin(a, f, rho)
    else:
        k1 = 1. / ((1. + (rho / eps)) * (1. + (rho2 / rho)))
        k2 = 1. / ((1. + (rho2 / eps)) * (1. + (rho / rho2)))
        xi1 = rho2 / (rho * (1. + (rho / eps) + (rho2 / eps)))
        xi2 = rho / (rho2 * (1. + (rho / eps) + (rho2 / eps)))
        g = aprox(sinkx(C, f, a, eps), eps, rho) - k2 * softmin(a, f, rho)
        g = g + xi2 * softmin(b, rho2, g)
        f = aprox(sinky(C, g, b, eps), eps, rho) - k1 * softmin(b, g, rho2)
        f = f + xi1 * softmin(a, rho, f)

    # Update on lambda
    t = rescale_potentials(f, g, a, b, rho, rho2)
    return f + t, g - t


def g_sinkhorn_loop(f, g, t, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g - t, eps, rho2) + t

    t = rescale_potentials(f, g, a, b, rho, rho2)

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f + t, eps, rho) - t

    # Update on lambda
    t = rescale_potentials(f, g, a, b, rho, rho2)

    return f, g, t
