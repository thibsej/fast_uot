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