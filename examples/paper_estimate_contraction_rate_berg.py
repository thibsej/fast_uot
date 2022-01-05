import numpy as np
import matplotlib.pyplot as plt
import os

from fastuot.numpy_berg import f_sinkhorn_loop, g_sinkhorn_loop, \
    h_sinkhorn_loop
from utils_examples import generate_synthetic_measure

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "/paper/"):
    os.mkdir(path + "/paper/")
if not os.path.isdir(path + "/rateberg/"):
    os.mkdir(path + "/rateberg/")

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True,
      'text.latex.preamble': [r'\usepackage{amsmath}',
                              r'\usepackage{amssymb}']}
plt.rcParams.update(rc)


def gauss(grid, mu, sig):
    return np.exp(-0.5 * ((grid-mu) / sig) ** 2)


if __name__ == '__main__':
    compute_data = False

    eps_l = [-1., 0.]
    N = 50
    a, x, b, y = generate_synthetic_measure(N, N)
    C = (x[:, None] - y[None, :])**2
    rho_scale = np.arange(-3., 3.5, 0.5)
    colors = [(1., 0., 0., 1.), (.5, 0., .5, 1.), (0., 0., 1., 1.)]

    string_method = ['f', 'g', 'h']
    func_method = [f_sinkhorn_loop, g_sinkhorn_loop, h_sinkhorn_loop]

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        np.save(path + "/rateberg/" + f"rho_scale.npy", rho_scale)
        for r in range(len(eps_l)):
            epst = 10**eps_l[r]
            rate = [[], [], []]
            for s in rho_scale:
                rhot = 10**s
                print(f"(eps, rho) = {(epst, rhot)}")

                # Compute reference
                fr, gr = np.zeros_like(a), np.zeros_like(b)
                for i in range(50000):
                    f_tmp = fr.copy()
                    if epst <= rhot:
                        fr, gr = g_sinkhorn_loop(fr, a, b, C, epst, rhot)
                    else:
                        fr, gr = f_sinkhorn_loop(fr, a, b, C, epst, rhot)

                    if np.amax(np.abs(fr - f_tmp)) < 1e-15:
                        break

                # Compute error and estimate rate
                for k, (s, loop) in enumerate(zip(string_method, func_method)):
                    err = []
                    f, g = np.zeros_like(a), np.zeros_like(b)
                    for i in range(2000):
                        f_tmp = f.copy()
                        f, g = loop(f, a, b, C, epst, rhot)

                        # t = 0.
                        # if s == 'h':
                        #     t = rescale_berg(f, g, a, b, rhot)
                        err.append(np.amax(np.abs(f - fr)))
                        if np.amax(np.abs(f - fr)) < 1e-12:
                            break
                    err = np.log10(np.array(err))
                    err = err[1:] - err[:-1]
                    rate[k].append(np.median(err))

            for k, (s, loop) in enumerate(zip(string_method, func_method)):
                np.save(
                    path + "/rateberg/" + "rate_" + s + f"_sinkhorn_berg_eps{epst}.npy",
                    np.array(rate[k]))

            #     # Compute error for F - Sinkhorn
            #     err_s = []
            #     f, g = np.zeros_like(a), np.zeros_like(b)
            #     for i in range(5000):
            #         f_tmp = f.copy()
            #         f, g = f_sinkhorn_loop(f, a, b, C, epst, rhot)
            #         err_s.append(np.amax(np.abs(f - fr)))
            #         if np.amax(np.abs(f - fr)) < 1e-12:
            #             break
            #     err_s = np.log10(np.array(err_s))
            #     err_s = err_s[1:] - err_s[:-1]
            #     rate_s.append(np.median(err_s))
            #
            #     # Compute error for G - Sinkhorn
            #     err_ti = []
            #     f, g = np.zeros_like(a), np.zeros_like(b)
            #     for i in range(5000):
            #         f_tmp = f.copy()
            #         f, g = g_sinkhorn_loop(f, a, b, C, epst, rhot)
            #         err_ti.append(np.amax(np.abs(f - fr)))
            #         if np.amax(np.abs(f - fr)) < 1e-12:
            #             break
            #     err_ti = np.log10(np.array(err_ti))
            #     err_ti = err_ti[1:] - err_ti[:-1]
            #     rate_ti.append(np.median(err_ti))
            #
            # # Plot results
            # np.save(path + "/rateberg/" + f"rate_f_sinkhorn_berg_eps{epst}.npy",
            #         np.array(rate_s))
            # np.save(path + "/rateberg/" + f"rate_g_sinkhorn_berg_eps{epst}.npy",
            #         np.array(rate_ti))

    ###########################################################################
    # Make plots
    ###########################################################################
    p = 0.97
    colors = ['cornflowerblue', 'indianred']
    markers = ['x', 'o', 'v']
    markevery = 2
    f, ax = plt.subplots(1, 1, figsize=(p * 5, p * 4))

    rho_scale = np.load(path + "/rateberg/" + f"rho_scale.npy")
    for lg_eps, color in zip(eps_l, colors):
        epst = 10**lg_eps
        rate_f = np.load(path + "/rateberg/" + f"rate_f_sinkhorn_berg_eps{epst}.npy")
        rate_g = np.load(path + "/rateberg/" + f"rate_g_sinkhorn_berg_eps{epst}.npy")
        rate_h = np.load(path + "/rateberg/" + f"rate_h_sinkhorn_berg_eps{epst}.npy")
        plt.plot(10**rho_scale, 10**rate_f, c=color, linestyle='dashed',
                 label='$\mathcal{F},\,\epsilon=$' + f' {epst}',
                 marker=markers[0],
                     markevery=markevery)
        plt.plot(10**rho_scale, 10**rate_g, c=color,
                 label='$\mathcal{F},\,\epsilon=$' + f' {epst}',
                 marker=markers[1],
                     markevery=markevery)
        plt.plot(10**rho_scale, 10**rate_h, c=color,
                 label='$\mathcal{F},\,\epsilon=$' + f' {epst}',
                 marker=markers[2],
                     markevery=markevery)
    ax.set_xlabel('Marginal parameter $\\rho$', fontsize=15)
    ax.set_title('Berg entropy', fontsize=18)
    ax.set_ylabel('Contraction rate', fontsize=15)
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12, ncol=3)
    plt.tight_layout()
    plt.savefig(path + "/paper/" + 'plot_log_contraction_berg.pdf')
    plt.show()