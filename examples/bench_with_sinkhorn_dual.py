import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from fastuot.uot1d import rescale_potentials, init_greed_uot, dual_loss, \
    solve_ot


def sinkx(C, f, a, eps):
    return - eps * logsumexp(np.log(a)[:,None] + (f[:,None] - C) / eps, axis=0)


def sinky(C, g, b, eps):
    return - eps * logsumexp(np.log(b)[None,:] + (g[None,:] - C) / eps, axis=1)


def aprox(f, eps, rho):
    return (1. / (1. + (eps / rho))) * f


def dual_score_ent(f, g, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho

    def phi(x, s):
        return -s * (np.exp(-x / s) - 1)

    return np.sum(a * phi(f, rho)) + np.sum(b * phi(g, rho2)) + np.sum(
        a[:, None] * b[None, :] * phi(C - f[:, None] - g[None, :], eps))


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


if __name__ == '__main__':
    # np.random.seed(0)
    N, M = 10, 11
    a, b = np.random.exponential(size=N), np.random.exponential(size=M)
    a, b = a / np.sum(a), b / np.sum(b)
    x, y = np.sort(np.random.normal(size=N)), np.sort(np.random.normal(size=M))


    rho = .1
    p = 2.
    nits = 500
    nbeg = 20
    C = np.abs(x[:, None] - y[None, :]) ** p

    # Bench FW-UOT
    # f, g = init_greed_uot(a, b, x, y, p, rho, rho2=None)
    # print("Dual gap", np.amin(np.abs(x[:, None] - y[None, :]) ** p
    #                           - f[:, None] - g[None, :]))
    f, g = np.zeros_like(a), np.zeros_like(b)
    dual_val = []
    for k in range(nits):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        A = np.exp(-f / rho) * a
        B = np.exp(-g / rho) * b
        dual_val.append(dual_loss(f, g, a, b, rho, rho2=rho))

        # update
        I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
        gamma = 2. / (2. + k)  # fixed decaying weights
        f = (1 - gamma) * f + gamma * fs
        g = (1 - gamma) * g + gamma * gs

    plt.plot(dual_val[nbeg:], label='fw')
    # print("\nDual pot for FW", f)
    # print("Dual gap", np.amin(np.abs(x[:, None] - y[None, :]) ** p
    #                           - f[:, None] - g[None, :]))

    # Bench Sinkhorn
    for eps in [1e-5]:
        f, g = np.zeros_like(a), np.zeros_like(b)
        dual_val = []
        for k in range(nits):
            f, g = sinkhorn_loop(f, a, b, C, eps, rho, rho2=None)
            dual_val.append(dual_loss(f, g, a, b, rho, rho2=rho))
            # dual_val.append(dual_score_ent(f, g, a, b, C, eps, rho, rho2=None))


        plt.plot(dual_val[nbeg:], label=f'S {eps}')
        # print(f"\nDual pot for Sinkh = {eps}", f)
        fpt, _ = sinkhorn_loop(f, a, b, C, eps, rho, rho2=None)
        print(np.amax(np.abs(f - fpt)))

    # Bench Sinkhorn
    for eps in [1e-5]:
        f, g = np.zeros_like(a), np.zeros_like(b)
        dual_val = []
        for k in range(nits):
            f, g = homogeneous_loop(f, a, b, C, eps, rho, rho2=None)
            dual_val.append(dual_loss(f, g, a, b, rho, rho2=rho))
            # dual_val.append(dual_score_ent(f, g, a, b, C, eps, rho, rho2=None))


        plt.plot(dual_val[nbeg:], label=f'S+T {eps}')
        # print(f"\nDual pot for Sinkh = {eps}", f)
        fpt, _ = sinkhorn_loop(f, a, b, C, eps, rho, rho2=None)
        print(np.amax(np.abs(f - fpt)))

    plt.legend()
    plt.show()
