import torch


def sinkx(C, f, a, eps):
    return - eps * (a.log()[:, None] + (f[:, None] - C) / eps).logsumexp(dim=0)


def sinky(C, g, b, eps):
    return - eps * (b.log()[None, :] + (g[None, :] - C) / eps).logsumexp(dim=1)


def softmin(a, f, rho):
    return - rho * (a.log() - f / rho).logsumexp(dim=0)


def aprox(f, eps, rho):
    return (1. / (1. + (eps / rho))) * f


def dual_score_ent(f, g, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho

    def phi(x, s):
        return -s * ((-x / s).exp() - 1)

    return (a * phi(f, rho)).sum() + (b * phi(g, rho2)).sum() \
           + (a[:, None] * b[None, :] * phi(C - f[:, None] - g[None, :],
                                            eps)).sum()


def rescale_potentials(f, g, a, b, rho, rho2):
    tau = (rho * rho2) / (rho + rho2)
    transl = tau * ((a.log() - f / rho).logsumexp(dim=0) -
                    (b.log() - g / rho2).logsumexp(dim=0))
    return transl


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
#
# def shift(f, g, a, b, C, eps, rho, rho2=None):
#     if rho2 is None:
#         rho2 = rho
#     xa = (a * (-f / rho).exp()).sum()
#     xb = (b * (-g / rho).exp()).sum()
#     xc = (a[:, None] * b[None, :] * (
#             (f[:, None] + g[None, :] - C) / eps).exp()).sum()
#     if rho2 == rho:
#         # Closed form when rho=rho2
#         k = (eps * rho) / (eps + 2 * rho)
#         return k * ((xa + xb).log() - torch.log(2) - xc.log())
#     else:
#         # Newton step
#         t = 0.0
#
#         def grad(t):
#             return xa * (-t / rho).exp() + xb * (
#                     -t / rho2).exp() - 2 * xc * (2 * t / eps).exp()
#
#         def hess(t):
#             return - xa * (-t / rho).exp() / rho - xb * (
#                     -t / rho2).exp() / rho2 - 4 * xc * (
#                            2 * t / eps).exp() / eps
#
#         for k in range(3):
#             t = t + grad(t) / hess(t)
#         return t
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
#     f, g = f + ts + tr, g + ts - tr
#
#     return f, g
