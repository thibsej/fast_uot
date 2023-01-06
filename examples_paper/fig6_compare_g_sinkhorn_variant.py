import numpy as np
import matplotlib.pyplot as plt
import os

from utils_examples import generate_synthetic_measure
from fastuot.numpy_sinkhorn import sinkx, sinky, aprox, rescale_potentials

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "gsink/"
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True,
      'text.latex.preamble': [r'\usepackage{amsmath}',
                              r'\usepackage{amssymb}']}
plt.rcParams.update(rc)


def g_sinkhorn_loop(f, g, t, a, b, C, eps, rho, rho2=None):
    if rho2 is None:
        rho2 = rho
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g - t, eps, rho2) + t

    t = rescale_potentials(f, g, a, b, rho, rho2)

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f + t, eps, rho) - t

    # Update on lambda
    t = rescale_potentials(f, g, a, b, rho, rho2)

    return f, g, t


if __name__ == '__main__':
    compute_data = False  # If false then load precomputed results and plots
    wot_data = True  # If true uses the WOT package biological data

    marginal_penalty_l = ['kl']
    penalty = marginal_penalty_l[0]
    from fastuot.numpy_sinkhorn import f_sinkhorn_loop, h_sinkhorn_loop

    # load data for computations
    N = 50
    a, x, b, y = generate_synthetic_measure(N, N+1)
    C = (x[:, None] - y[None, :]) ** 2
    dataname = 'synth'

    # Grid of parameters for Sinkhorn algorithm
    eps_l = [np.log10(0.1)]
    rho_scale = np.arange(-3., 2.5, 0.5)
    string_method = ['f', 'g', 'h']
    func_method = [f_sinkhorn_loop, g_sinkhorn_loop, h_sinkhorn_loop]

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        np.save(path + f"rho_scale.npy", rho_scale)
        for r in range(len(eps_l)):
            epst = 10 ** eps_l[r]
            rate = [[], [], []]
            for s in rho_scale:
                rhot = 10 ** s
                print(f"(eps, rho) = {(epst, rhot)}")

                # Compute reference
                fr, gr = np.zeros_like(a), np.zeros_like(b)
                for i in range(500):
                    f_tmp = fr.copy()
                    fr, gr = h_sinkhorn_loop(fr, a, b, C, epst, rhot)
                    fr, gr = f_sinkhorn_loop(fr, a, b, C, epst, rhot)
                    if np.amax(np.abs(fr - f_tmp)) < 1e-15:
                        break
                print(f"     Estimated fixed point: Iter {i}")

                # Compute error and estimate rate
                for k, (s, loop) in enumerate(zip(string_method, func_method)):
                    err = []
                    if s == 'g':
                        f, g, t = np.zeros_like(a), np.zeros_like(b), 0.
                        for i in range(100):
                            f_tmp = f.copy()
                            f, g, t = g_sinkhorn_loop(f, g, t, a, b, C,
                                                      epst, rhot, rho2=None)

                            err.append(np.amax(np.abs(f + t - fr)))
                            if np.amax(np.abs(f + t - fr)) < 1e-13:
                                break
                    else:
                        f, g = np.zeros_like(a), np.zeros_like(b)
                        for i in range(100):
                            f_tmp = f.copy()
                            f, g = loop(f, a, b, C, epst, rhot)

                            err.append(np.amax(np.abs(f - fr)))
                            if np.amax(np.abs(f - fr)) < 1e-13:
                                break

                    err = np.log10(np.array(err))
                    err = err[1:] - err[:-1]
                    rate[k].append(np.median(err))

            for k, (s, loop) in enumerate(zip(string_method, func_method)):
                np.save(
                    path + "rate_" + s + f"_sinkhorn_{penalty}_eps{epst}_{dataname}.npy",
                    np.array(rate[k]))

    ###########################################################################
    # Make plots
    ###########################################################################
    p = 0.9
    colors = ['cornflowerblue', 'mediumseagreen', 'indianred']
    markers = ['x', 'v', 'o']
    linestyles = ['dotted', 'solid', 'dashed']
    labels = ['S, $\epsilon=$', '$\mathcal{G},\,\epsilon=$',
              'TI, $\epsilon=$']
    markevery = 2
    f, ax = plt.subplots(1, 1, figsize=(p * 6, p * 4))

    rho_scale = 10 ** np.load(path + f"rho_scale.npy")

    for logeps in eps_l:
        epst = 10 ** logeps
        for linestyle, label, s, color, marker in zip(linestyles, labels,
                                                      string_method, colors,
                                                      markers):
            rate_f = np.load(
                path + f"rate_" + s + f"_sinkhorn_{penalty}_eps{epst}_{dataname}.npy")
            ax.plot(rho_scale, 10 ** rate_f, c=color, linestyle=linestyle,
                    label=label + f' {np.around(epst, decimals=1)}',
                    marker=marker, markevery=markevery)

    ax.legend(fontsize=11, ncol=1, columnspacing=0.5, handlelength=2.,
              loc=(.7, .02))

    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Marginal parameter $\\rho$', fontsize=18)
    ax.set_title('KL entropy', fontsize=22)
    ax.set_ylabel('Contraction rate', fontsize=18)

    plt.tight_layout()
    plt.savefig(path + f'plot_log_contraction_rate_gsink_{penalty}_{dataname}.pdf')
    plt.show()