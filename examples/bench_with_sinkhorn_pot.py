import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import cvxpy as cp
import time as time
import os

from fastuot.uot1d import rescale_potentials, solve_ot

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_comparison/"
if not os.path.isdir(path):
    os.mkdir(path)


def dual_via_cvxpy(a, b, x, y, p, rho):
    C = np.abs(x[:, None] - y[None, :]) ** p
    f = cp.Variable((x.shape[0]))
    g = cp.Variable((y.shape[0]))
    inta = cp.sum(cp.multiply(a, cp.exp(-f / rho)))
    intb = cp.sum(cp.multiply(b, cp.exp(-g / rho)))
    constr = [f[:, None] + g[None, :] <= C]
    objective = cp.Minimize(inta + intb)
    prob = cp.Problem(objective, constr)
    result = prob.solve(max_iters=50000, verbose=False,
                        solver=cp.SCS, eps=1e-7)
    # result = prob.solve(max_iters=30000, verbose=False,
    #                     solver=cp.ECOS, abstol=1e-7)
    return result, constr, f, g


def sinkx(C, f, a, eps):
    return - eps * logsumexp(np.log(a)[:, None] + (f[:, None] - C) / eps,
                             axis=0)


def sinky(C, g, b, eps):
    return - eps * logsumexp(np.log(b)[None, :] + (g[None, :] - C) / eps,
                             axis=1)


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


def rescale_exp(u, v, a, b, rho, eps):
    A = np.sum(a * u ** (-rho / eps))
    B = np.sum(b * v ** (-rho / eps))
    t = (A / B) ** (0.5 * rho / eps)
    return t


def fast_homogeneous_loop(u, a, b, K, eps, rho):
    # Update on G
    v = 1. / K.T.dot(u * a)
    v = v ** (rho / (rho + eps))
    t = rescale_exp(u, v, a, b, rho, eps)
    v = v / t

    # Update on F
    u = 1. / K.dot(v * b)
    u = u ** (rho / (rho + eps))
    t = rescale_exp(u, v, a, b, rho, eps)
    u = u * t

    return u, v


if __name__ == '__main__':
    Ntrials = 20
    rho = 10.
    p = 2.
    nits = 5000
    nbeg = 0
    N, M = 50, 51
    list_eps_lse = [-2., -2.5, -3., -3.5, -4.]
    acc_fw = np.zeros((Ntrials, nits))
    acc_lse = np.zeros((len(list_eps_lse), Ntrials, nits))
    time_fw = np.zeros((Ntrials, nits))
    time_lse = np.zeros((len(list_eps_lse), Ntrials, nits))
    # list_eps_exp = [1e-1]
    # acc_exp = np.zeros((len(list_eps_exp), Ntrials, nits))
    # time_exp = np.zeros((len(list_eps_exp), Ntrials, nits))

    for i in range(Ntrials):
        print(f"Trial {i}")
        np.random.seed(i)
        a, b = np.random.exponential(size=N), np.random.exponential(size=M)
        a, b = a / np.sum(a), b / np.sum(b)
        x, y = np.sort(np.random.normal(size=N)), \
               np.sort(np.random.normal(loc=0.2, size=M))
        C = np.abs(x[:, None] - y[None, :]) ** p

        _, _, f, _ = dual_via_cvxpy(a, b, x, y, p, rho)
        fr = f.value
        print("Computed reference potential")

        # Bench FW-UOT
        f, g = np.zeros_like(a), np.zeros_like(b)
        dual_val = []
        for k in range(nits):
            t0 = time.time()
            transl = rescale_potentials(f, g, a, b, rho, rho)
            f, g = f + transl, g - transl
            A = np.exp(-f / rho) * a
            B = np.exp(-g / rho) * b

            # update
            I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
            gamma = 2. / (2. + k)  # fixed decaying weights
            f = (1 - gamma) * f + gamma * fs
            g = (1 - gamma) * g + gamma * gs
            acc_fw[i, k] = np.log10(np.amax(np.abs(f - fr)))
            time_fw[i, k] = time.time() - t0

        # Bench LSE Sinkhorn with translation
        for j in range(len(list_eps_lse)):
            eps = 10 ** list_eps_lse[j]
            f, g = np.zeros_like(a), np.zeros_like(b)
            dual_val = []
            for k in range(nits):
                t0 = time.time()
                f, g = homogeneous_loop(f, a, b, C, eps, rho, rho2=None)
                time_lse[j, i, k] = time.time() - t0
                acc_lse[j, i, k] = np.log10(np.amax(np.abs(f - fr)))

            fpt, _ = sinkhorn_loop(f, a, b, C, eps, rho, rho2=None)
            print(np.amax(np.abs(f - fpt)))

        # for j in range(len(list_eps_exp)):
        #     eps = list_eps_exp[j]
        #     K = np.exp(-C / eps)
        #     u, v = np.ones_like(a), np.ones_like(b)
        #     dual_val = []
        #     for k in range(nits):
        #         t0 = time.time()
        #         u, v = fast_homogeneous_loop(u, a, b, K, eps, rho)
        #         time_exp[j, i, k] = time.time() - t0
        #         f = eps * np.log(u)
        #         acc_exp[j, i, k] = np.log10(np.amax(np.abs(f - fr)))

    plt.plot(np.mean(time_fw[:, :]) * np.arange(nits),
                 np.mean(acc_fw, axis=0), c='b', label='fw')
    cmap = plt.get_cmap('autumn')
    for j in range(len(list_eps_lse)):
        k = list_eps_lse[j]
        eps = 10 ** k
        plt.plot(np.mean(time_lse[j, :, :]) * np.arange(nits),
                     np.mean(acc_lse[j, :, :], axis=0),
                     label=f'S+T {k}',
                     c=cmap(j / len(list_eps_lse)))
    plt.legend()
    plt.xlabel('time', fontsize=16)
    plt.ylabel('$\log||f_t - f_*||_\infty$', fontsize=16)
    plt.tight_layout()
    plt.savefig(path + f"bench_comparison_fast_sink_fw_rho{rho}.png")
    plt.show()
