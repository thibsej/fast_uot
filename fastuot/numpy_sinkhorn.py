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


def g_sinkhorn_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)

    # Update on lambda
    t = rescale_potentials(f, g, a, b, rho, rho2)

    return f + t, g - t


def h_sinkhorn_loop(f, a, b, C, eps, rho):
    g = aprox(sinkx(C, f, a, eps), eps, rho) \
        - 0.5 * (eps / (eps + rho)) * softmin(a, f, rho)
    g = g + (eps / (eps + 2 * rho)) * softmin(b, g, rho)
    f = aprox(sinky(C, g, b, eps), eps, rho) \
        - 0.5 * (eps / (eps + rho)) * softmin(b, g, rho)
    f = f + (eps / (eps + 2 * rho)) * softmin(a, f, rho)
    return f, g


###############################################################################
# Deprecated code
###############################################################################

# def shift(f, g, a, b, C, eps, rho, rho2=None):
#     """Computes the optimal translation (f+t, g+t)"""
#     if rho2 is None:
#         rho2 = rho
#     xa = np.sum(a * np.exp(-f / rho))
#     xb = np.sum(b * np.exp(-g / rho2))
#     xc = np.sum(
#         a[:, None] * b[None, :] * np.exp((f[:, None] + g[None, :] - C) / eps))
#     if rho2 is None:
#         # Closed form when rho=rho2
#         k = (eps * rho) / (eps + 2 * rho)
#         return k * (np.log(xa + xb) - np.log(2) - np.log(xc))
#     else:
#         # Newton step
#         t = 0.0
#
#         def grad(t):
#             return xa * np.exp(-t / rho) + xb * np.exp(
#                 -t / rho2) - 2 * xc * np.exp(2 * t / eps)
#
#         def hess(t):
#             return - xa * np.exp(-t / rho) / rho - xb * np.exp(
#                 -t / rho2) / rho2 - 4 * xc * np.exp(2 * t / eps) / eps
#
#         for k in range(3):
#             t = t + grad(t) / hess(t)
#         return t
#
#
# def homogeneous_loop2(f, a, b, C, eps, rho, rho2=None):
#     if rho2 is None:
#         rho2 = rho
#     # Update on G
#     g = sinkx(C, f, a, eps)
#     g = aprox(g, eps, rho2)
#     t = rescale_potentials(f, g, a, b, rho, rho2)
#     g = g - t
#
#     # Update on F
#     f = sinky(C, g, b, eps)
#     f = aprox(f, eps, rho)
#     t = rescale_potentials(f, g, a, b, rho, rho2)
#     f = f + t
#
#     return f, g
#
#
# def shift_loop(f, a, b, C, eps, rho, rho2=None):
#     if rho2 is None:
#         rho2 = rho
#     # Update on G
#     g = sinkx(C, f, a, eps)
#     g = aprox(g, eps, rho2)
#     t = shift(f, g, a, b, C, eps, rho, rho2)
#     g = g + t
#
#     # Update on F
#     f = sinky(C, g, b, eps)
#     f = aprox(f, eps, rho)
#     t = shift(f, g, a, b, C, eps, rho, rho2)
#     f = f + t
#
#     return f, g
#
#
# def full_loop(f, a, b, C, eps, rho, rho2=None):
#     if rho2 is None:
#         rho2 = rho
#     # Update on G
#     g = sinkx(C, f, a, eps)
#     g = aprox(g, eps, rho2)
#     ts = shift(f, g, a, b, C, eps, rho, rho2)
#     tr = rescale_potentials(f + ts, g + ts, a, b, rho, rho2)
#     g = g + ts - tr
#
#     # Update on F
#     f = sinky(C, g, b, eps)
#     f = aprox(f, eps, rho)
#     ts = shift(f, g, a, b, C, eps, rho, rho2)
#     tr = rescale_potentials(f + ts, g + ts, a, b, rho, rho2)
#     f = f + ts + tr
#
#     return f, g
#
#
# def rescale_exp(u, v, a, b, rho, eps):
#     A = np.sum(a * u ** (-rho / eps))
#     B = np.sum(b * v ** (-rho / eps))
#     t = (A / B) ** (0.5 * rho / eps)
#     return t
#
#
# def fast_homogeneous_loop(u, a, b, K, eps, rho):
#     # Update on G
#     v = 1. / K.T.dot(u * a)
#     v = v ** (rho / (rho + eps))
#     t = rescale_exp(u, v, a, b, rho, eps)
#     v = v / t
#
#     # Update on F
#     u = 1. / K.dot(v * b)
#     u = u ** (rho / (rho + eps))
#     t = rescale_exp(u, v, a, b, rho, eps)
#     u = u * t
#
#     return u, v
