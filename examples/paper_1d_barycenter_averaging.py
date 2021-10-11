import os

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import progressbar

from fastuot.uot1dbar import solve_unbalanced_barycenter, \
    solve_balanced_barycenter

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/paper/"
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)


def parzen_window(y, P, s=.005, grid=np.linspace(0, 1, 400)):
    W = np.exp(-(y[None, :] - grid[:, None]) ** 2 / (2 * s ** 2))
    W = W / np.sum(W, axis=0)[None, :]
    return W @ P


def setup_axes(P):
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.axis([0, 1, 0, 1.1 * max(P)])


def gauss(t, mu, sig):
    return np.exp(-(t-mu)**2/(2*sig**2))


def generate_mixtures(K, nsampl, sig):
    t = np.linspace(0, 1, nsampl)
    x, a = [], []
    for k in range(K):
        x.append(t)
        u, v = .1 + .3 * np.random.rand(), .6 + .3 * np.random.rand()
        g, h = .8 + .2 * np.random.rand(), .8 + .2 * np.random.rand()
        # g, h = 0.5, 0.5
        b = g * gauss(t, u, sig) + h * gauss(t, v, sig)
        a.append(b / np.sum(b))
    # ref = gauss(t, 0.25, sig) + gauss(t, 0.75, sig)
    # ref = ref / np.sum(ref)
    return a, x


if __name__ == '__main__':
    m = 1000  # size of the grid
    grid_pw = np.linspace(0, 1, m)
    K = 10
    nsampl = 500
    sig = .03
    np.random.seed(0)

    a, x = generate_mixtures(K, nsampl, sig)
    lam = np.ones(K) / K

    # Plot input measures
    plt.figure(figsize=(7, 6))
    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.fill_between(x[k], a[k], alpha=1)
        setup_axes(a[k])
    plt.tight_layout()
    plt.savefig(path + 'plot_inputs_barycenter.pdf', bbox_inches='tight')
    plt.show()

    # Compute balanced barycenter and plot
    I, P, y, f, cost = solve_balanced_barycenter(a, x, lam)
    b = parzen_window(y, P, grid=grid_pw)
    plt.figure(figsize=(7, 6))
    plt.fill_between(grid_pw, b, 'k')
    setup_axes(b)
    plt.tight_layout()
    plt.savefig(path + 'plot_balanced_barycenter.pdf')
    plt.show()

    # Compute unbalanced barycenter
    rho = .3
    Iu,Pu,yu,fu = solve_unbalanced_barycenter(a, x, lam, rho,
                                              niter=500, verb=True)
    ub = parzen_window(yu, Pu, grid=grid_pw)
    plt.figure(figsize=(7, 6))
    plt.plot(grid_pw, b, 'b:', label='$balanced$')
    plt.fill_between(grid_pw, ub, color='r', label='$unbalanced$')
    plt.legend(loc=9, fontsize=16)
    setup_axes(b)
    plt.tight_layout()
    plt.savefig(path + 'plot_unbalanced_barycenter.pdf')
    plt.show()



