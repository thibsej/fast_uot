import torch


def sinkx(C, f, a, eps):
    return - eps * (a.log()[:, None] + (f[:, None] - C) / eps).logsumexp(dim=0)


def sinky(C, g, b, eps):
    return - eps * (b.log()[None, :] + (g[None, :] - C) / eps).logsumexp(dim=1)


def aprox(f, eps, rho):
    return (1. / (1. + (eps / rho))) * f


def dual_score_ent(f, g, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho

    def phi(x, s):
        return -s * ((-x / s).exp() - 1)

    return (a * phi(f, rho)).sum() + sum(b * phi(g, rho2)).sum() \
           + (a[:, None] * b[None, :] * phi(C - f[:, None] - g[None, :],
                                            eps)).sum()


def rescale_potentials(f, g, a, b, rho, rho2):
    tau = (rho * rho2) / (rho + rho2)
    transl = tau * ((a.log() - f / rho).logsumexp(dim=0) -
                    (b.log() - g / rho2).logsumexp(dim=0))
    return transl


def shift(f, g, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    xa = (a * (-f / rho).exp()).sum()
    xb = (b * (-g / rho).exp()).sum()
    xc = (a[:, None] * b[None, :] * ((f[:, None] + g[None, :] - C) / eps).exp()).sum()
    if rho2 == rho:
        # Closed form when rho=rho2
        k = (eps * rho) / (eps + 2 * rho)
        return k * ((xa + xb).log() - torch.log(2) - xc.log())
    else:
        # Newton step
        t = 0.0

        def grad(t):
            return xa * (-t / rho).exp() + xb * (
                -t / rho2).exp() - 2 * xc * (2 * t / eps).exp()

        def hess(t):
            return - xa * (-t / rho).exp() / rho - xb * (
                -t / rho2).exp() / rho2 - 4 * xc * (2 * t / eps).exp() / eps

        for k in range(3):
            t = t + grad(t) / hess(t)
        return t


def sinkhorn_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)
    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)
    return f, g


def homogeneous_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)
    t = rescale_potentials(f, g, a, b, rho, rho2)
    g = g - t

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)
    t = rescale_potentials(f, g, a, b, rho, rho2)
    f = f + t

    return f, g


def shift_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)
    t = shift(f, g, a, b, C, eps, rho, rho2)
    g = g + t

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)
    t = shift(f, g, a, b, C, eps, rho, rho2)
    f = f + t

    return f, g


def full_loop(f, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)
    ts = shift(f, g, a, b, C, eps, rho, rho2)
    tr = rescale_potentials(f + ts, g + ts, a, b, rho, rho2)
    g = g + ts - tr

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)
    ts = shift(f, g, a, b, C, eps, rho, rho2)
    tr = rescale_potentials(f + ts, g + ts, a, b, rho, rho2)
    f, g = f + ts + tr, g + ts -tr

    return f, g
