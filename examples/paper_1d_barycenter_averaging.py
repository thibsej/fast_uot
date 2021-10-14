import os

import numpy as np
import matplotlib.pyplot as plt

from fastuot.uot1dbar import solve_unbalanced_barycenter, \
    solve_balanced_barycenter

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "/paper/"):
    os.mkdir(path + "/paper/")
if not os.path.isdir(path + "/uot_barycenter/"):
    os.mkdir(path + "/uot_barycenter/")

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
    return np.exp(-(t - mu) ** 2 / (2 * sig ** 2))


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
    compute_data = True

    m = 1000  # size of the grid
    grid_pw = np.linspace(0, 1, m)
    K = 8
    lam = np.ones(K) / K
    nsampl = 500
    sig = .03
    rho = .3
    niter_uot_fw = 1000
    np.random.seed(0)

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        a, x = generate_mixtures(K, nsampl, sig)
        for k in range(len(a)):
            meas = np.zeros((2, nsampl))
            meas[0], meas[1] = a[k], x[k]
            np.save(path + "/uot_barycenter/" + f"input_measure_{k}.npy", meas)

        I, P, y, f, _ = solve_balanced_barycenter(a, x, lam)
        np.save(path + "/uot_barycenter/" + f"support_balanced_bar.npy", y)
        np.save(path + "/uot_barycenter/" + f"weights_balanced_bar.npy", P)

        # Compute unbalanced barycenter
        Iu, Pu, yu, fu, cost = solve_unbalanced_barycenter(a, x, lam, rho,
                                                           niter=niter_uot_fw,
                                                           verb=True)
        np.save(path + "/uot_barycenter/" + f"support_unbalanced_bar.npy", yu)
        np.save(path + "/uot_barycenter/" + f"weights_unbalanced_bar.npy", Pu)
        np.save(path + "/uot_barycenter/" + f"score_unbalanced_bar.npy",
                np.array(cost))

    ###########################################################################
    # Make plots
    ###########################################################################

    # Plot input measures
    plt.figure(figsize=(8, 5))
    for k in range(K):
        meas = np.load(path + "/uot_barycenter/" + f"input_measure_{k}.npy")
        plt.subplot(K, 1, k + 1)
        plt.fill_between(meas[1], meas[0], alpha=1)
        setup_axes(meas[0])
    plt.tight_layout()
    plt.savefig(path + 'plot_inputs_barycenter.pdf', bbox_inches='tight')
    plt.show()

    # Compute balanced barycenter and plot
    y = np.load(path + "/uot_barycenter/" + f"support_balanced_bar.npy")
    P = np.load(path + "/uot_barycenter/" + f"weights_balanced_bar.npy")
    b = parzen_window(y, P, grid=grid_pw)
    plt.figure(figsize=(8, 5))
    plt.fill_between(grid_pw, b, 'k')
    setup_axes(b)
    plt.tight_layout()
    plt.savefig(path + 'plot_balanced_barycenter.pdf')
    plt.show()

    # Compute unbalanced barycenter
    yu = np.load(path + "/uot_barycenter/" + f"support_unbalanced_bar.npy")
    Pu = np.load(path + "/uot_barycenter/" + f"weights_unbalanced_bar.npy")
    ub = parzen_window(yu, Pu, grid=grid_pw)
    plt.figure(figsize=(8, 5))
    plt.plot(grid_pw, b, 'b:', label='$balanced$')
    plt.fill_between(grid_pw, ub, color='r', label='$unbalanced$')
    plt.legend(loc=9, fontsize=16)
    setup_axes(b)
    plt.tight_layout()
    plt.savefig(path + 'plot_unbalanced_barycenter.pdf')
    plt.show()

    # Plot dual score
    cost = np.load(path + "/uot_barycenter/" + f"score_unbalanced_bar.npy")
    plt.plot(cost[1:])
    plt.show()