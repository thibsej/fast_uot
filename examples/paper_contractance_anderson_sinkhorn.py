import numpy as np
import matplotlib.pyplot as plt
import os

from fastuot.numpy_sinkhorn import f_sinkhorn_loop, h_sinkhorn_loop
from fastuot.uot1d import rescale_potentials
from utils_examples import generate_synthetic_measure

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + 'rateanderson/'
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True,
      'text.latex.preamble': [r'\usepackage{amsmath}',
                              r'\usepackage{amssymb}']}
plt.rcParams.update(rc)


def anderson_f_sinkhorn_loop(f, a, b, C, eps, rho, K=4, reg=1e-7):
    U = np.zeros((K + 1, f.shape[0]))
    U[0] = f
    for k in range(K):
        f, g = f_sinkhorn_loop(f, a, b, C, eps, rho)
        U[k + 1] = f
    L = U[1:, :] - U[:-1, :]
    L = L.dot(L.T)
    c = np.linalg.solve(L + reg * np.eye(K), np.ones(K))
    c = c / np.sum(c)
    f = c.dot(U[:-1, :])
    f, g = f_sinkhorn_loop(f, a, b, C, eps, rho)
    return f, g


def anderson_h_sinkhorn_loop(f, a, b, C, eps, rho, K=4, reg=1e-7):
    U = np.zeros((K + 1, f.shape[0]))
    U[0] = f
    for k in range(K):
        f, g = h_sinkhorn_loop(f, a, b, C, eps, rho)
        U[k + 1] = f
    L = U[1:, :] - U[:-1, :]
    L = L.dot(L.T)
    c = np.linalg.solve(L + reg * np.eye(K), np.ones(K))
    c = c / np.sum(c)
    f = c.dot(U[:-1, :])
    f, g = h_sinkhorn_loop(f, a, b, C, eps, rho)
    return f, g


def run_error_loop(loop_func, fr, epst, rhot):
    error = []
    f, g = np.zeros_like(a), np.zeros_like(b)
    for i in range(1000):
        f, g = loop_func(f, a, b, C, epst, rhot)
        t = 0.0
        if loop_func in [h_sinkhorn_loop, anderson_h_sinkhorn_loop]:
            t = rescale_potentials(f, g, a, b, rhot)
        error.append(np.amax(np.abs(f + t - fr)))
        if np.amax(np.abs(f + t - fr)) < 1e-12:
            break
    error = np.log10(np.array(error))
    error = error[1:] - error[:-1]
    return error


if __name__ == '__main__':
    compute_data = False

    eps_l = [-1.]
    N = 50
    a, x, b, y = generate_synthetic_measure(N, N)
    C = (x[:, None] - y[None, :]) ** 2
    rho_scale = np.arange(-3., 3.5, 0.5)
    list_loops = [f_sinkhorn_loop, h_sinkhorn_loop,
                  anderson_f_sinkhorn_loop, anderson_h_sinkhorn_loop]

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        np.save(path + f"rho_scale.npy", rho_scale)
        for r in range(len(eps_l)):
            epst = 10 ** eps_l[r]
            rate_f, rate_g, rate_h = [], [], []
            rate_andf, rate_andg, rate_andh = [], [], []
            list_rates = [rate_f, rate_g, rate_h,
                          rate_andf, rate_andg, rate_andh]
            list_fnames = [f"rate_f_sinkhorn_kl_eps{epst}.npy",
                           f"rate_h_sinkhorn_kl_eps{epst}.npy",
                           f"rate_andf_sinkhorn_kl_eps{epst}.npy",
                           f"rate_andhf_sinkhorn_kl_eps{epst}.npy"]
            for s in rho_scale:
                rhot = 10 ** s
                print(f"(eps, rho) = {(epst, rhot)}")

                # Compute reference
                fr, gr = np.zeros_like(a), np.zeros_like(b)
                for i in range(50000):
                    f_tmp = fr.copy()
                    fr, gr = h_sinkhorn_loop(fr, a, b, C, epst, rhot)
                    t = rescale_potentials(fr, gr, a, b, rhot)
                    fr, gr = fr + t, gr - t
                    if np.amax(np.abs(fr - f_tmp)) < 1e-15:
                        break
                print("computed reference.")

                for loop_func, rate in zip(list_loops, list_rates):
                    error = run_error_loop(loop_func, fr, epst, rhot)
                    rate.append(np.median(error))

            for rate, fname in zip(list_rates, list_fnames):
                np.save(path + fname, rate)

    ###########################################################################
    # Make plots
    ###########################################################################
    p = 0.97
    colors = ['cornflowerblue', 'indianred',
              'cornflowerblue', 'indianred']
    markers = ['x', 'v', 'x', 'v']
    linestyles = ['dotted', 'dotted', 'dashed', 'dashed']
    labels = ['S',
              'TI', 'S, And.',
              'TI, And.']
    markevery = 2
    f, ax = plt.subplots(1, 1, figsize=(p * 5, p * 4))

    rho_scale = 10 ** np.load(path + f"rho_scale.npy")

    for r in range(len(eps_l)):
        epst = 10 ** eps_l[r]
        list_fnames = [f"rate_f_sinkhorn_kl_eps{epst}.npy",
                       f"rate_h_sinkhorn_kl_eps{epst}.npy",
                       f"rate_andf_sinkhorn_kl_eps{epst}.npy",
                       f"rate_andhf_sinkhorn_kl_eps{epst}.npy"]
        list_rates = []
        for fname in list_fnames:
            list_rates.append(np.load(path + fname))

        for r in range(len(list_fnames)):
            ax.plot(rho_scale, 10 ** list_rates[r], c=colors[r],
                    linestyle=linestyles[r],
                    label=labels[r], marker=markers[r],
                    markevery=markevery)

    ax.legend(fontsize=11, ncol=2, columnspacing=0.5, handlelength=2.,
              loc=(.02, .02))

    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Marginal parameter $\\rho$', fontsize=18)
    ax.set_title('KL entropy', fontsize=22)
    ax.set_ylabel('Contraction rate', fontsize=18)

    plt.tight_layout()
    plt.savefig(path + 'plot_log_contraction_rate_anderson.pdf')
    plt.show()
