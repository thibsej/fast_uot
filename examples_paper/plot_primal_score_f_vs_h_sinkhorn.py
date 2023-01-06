"""
TODO: Store F sinkhorn and H sinkhorn potentials, then use primal dual
optimality of plan to compute primal score and compare both versions.
Compare (or not) in log-scale with approximated optimal
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from fastuot.numpy_sinkhorn import h_sinkhorn_loop, f_sinkhorn_loop, dual_score_ent
from utils_examples import generate_synthetic_measure

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "primalcost/"):
    os.mkdir(path + "primalcost/")
path = path + "primalcost/"


def kl_entropy(x):
    return x * np.log(x + 1e-16) - x + 1


def primal_cost(pi, a, b, eps, rho):
    pi1, pi2 = np.sum(pi, axis=1), np.sum(pi, axis=0)
    cost = rho * np.sum(a * kl_entropy(pi1 / a))
    cost = cost + rho * np.sum(b * kl_entropy(pi2 / b))
    cost = cost - eps * np.sum(pi) + eps * np.sum(a) * np.sum(b)
    return cost


if __name__ == '__main__':
    compute_data = True

    rho = 10.
    eps = 1e-4
    p = 2.
    nits = 1000
    N, M = 20, 21
    a, x, b, y = generate_synthetic_measure(N, M)
    C = np.abs(x[:, None] - y[None, :]) ** p

    if compute_data:
        # Compute exact potentaial
        fr = np.zeros_like(a)
        for i in range(50000):
            f_tmp = fr.copy()
            fr, gr = h_sinkhorn_loop(fr, a, b, C, eps, rho)
            if np.amax(np.abs(fr - f_tmp)) < 1e-14:
                break
        cost_r = dual_score_ent(fr, gr, a, b, C, eps, rho, rho2=None)


        # compute potential and primal cost
        f_f, f_h = np.zeros_like(a), np.zeros_like(a)
        cost_f, cost_h = np.zeros(nits), np.zeros(nits)
        for i in range(nits):
            f_f, g_f = f_sinkhorn_loop(f_f, a, b, C, eps, rho)
            f_h, g_h = h_sinkhorn_loop(f_h, a, b, C, eps, rho)

            pi_f = np.exp((f_f[:,None] + g_f[None:] - C) / eps) * a[:,None] * b[None,:]
            pi_h = np.exp((f_h[:, None] + g_h[None:] - C) / eps) * a[:,None] * b[None,:]
            cost_f[i] = primal_cost(pi_f, a, b, eps, rho) #- cost_r
            cost_h[i] = primal_cost(pi_h, a, b, eps, rho) #- cost_r
        np.save(path + "primal_cost_f.npy", cost_f)
        np.save(path + "primal_cost_h.npy", cost_h)


    ###########################################################################
    # Make plots
    ###########################################################################
    cost_f = np.load(path + "primal_cost_f.npy")
    cost_h = np.load(path + "primal_cost_h.npy")

    p = 0.97
    colors = ['cornflowerblue', 'indianred']
    markers = ['x', 'o']
    linestyles = ['dotted', 'dashed']
    labels = ['$\mathcal{F}$','$\mathcal{H}$']
    markevery = 2
    costs = [cost_f, cost_h]

    f, ax = plt.subplots(1, 1, figsize=(p * 5, p * 4))

    for marker, color, linestyle, label, cost in zip(markers, colors, linestyles, labels, costs):
        ax.plot(cost, c=color, linestyle=linestyle,
                label=label, marker=marker, markevery=markevery)
    # ax.plot(cost_f - cost_h)
    ax.legend(fontsize=11, ncol=2, columnspacing=0.5, handlelength=1.3,
              loc=(.4, .02))

    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Iterations', fontsize=15)
    ax.set_ylabel('Primal cost', fontsize=15)

    plt.tight_layout()
    plt.savefig(path + f'plot_primal_cost.pdf')
    plt.show()