import numpy as np
import matplotlib.pyplot as plt
import os

from fastuot.numpy_sinkhorn import sinkhorn_loop, faster_loop, balanced_loop
from fastuot.numpy_sinkhorn import homogeneous_loop as numpy_loop
from fastuot.uot1d import hilbert_norm, rescale_potentials

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "/paper/"):
    os.mkdir(path + "/paper/")
if not os.path.isdir(path + "/ratekl/"):
    os.mkdir(path + "/ratekl/")

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)


def gauss(grid, mu, sig):
    return np.exp(-0.5 * ((grid-mu) / sig) ** 2)


def normalize(x):
    return x / np.sum(x)


def generate_measure(N):
    x = np.linspace(0.2, 0.4, num=N)
    a = np.zeros_like(x)
    a[:N//2] = 2.
    a[N//2:] = 3.
    y = np.linspace(0.45, 0.95, num=N)
    a = normalize(a)
    b = normalize(gauss(y, 0.6, 0.03)
                  + gauss(y, 0.7, 0.03)
                  + gauss(y, 0.8, 0.03))
    return a, x, b, y

if __name__ == '__main__':
    compute_data = False

    eps_l = [-2., -1., 0.]
    N = 50
    a, x, b, y = generate_measure(N)
    C = (x[:, None] - y[None, :])**2
    rho_scale = np.arange(-3., 3.5, 0.5)
    colors = [(1., 0., 0., 1.), (.5, 0., .5, 1.), (0., 0., 1., 1.)]

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        np.save(path + "/ratekl/" + f"rho_scale.npy", rho_scale)
        for r in range(len(eps_l)):
            epst = 10 ** eps_l[r]
            rate_s, rate_ti, rate_f = [], [], []
            for s in rho_scale:
                rhot = 10**s
                print(f"(eps, rho) = {(epst, rhot)}")
                # Compute reference
                fr, gr = np.zeros_like(a), np.zeros_like(b)
                for i in range(50000):
                    f_tmp = fr.copy()
                    if epst <= rhot:
                        fr, gr = numpy_loop(fr, a, b, C, epst, rhot)
                    else:
                        fr, gr = sinkhorn_loop(fr, a, b, C, epst, rhot)
                    if np.amax(np.abs(fr - f_tmp)) < 1e-15:
                        break

                # Compute error for F - Sinkhorn
                err_s = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(5000):
                    f_tmp = f.copy()
                    f, g = sinkhorn_loop(f, a, b, C, epst, rhot)
                    t = rescale_potentials(f, g, a, b, rhot)
                    err_s.append(np.amax(np.abs(f + t - fr)))
                    if np.amax(np.abs(f + t - fr)) < 1e-12:
                        break
                err_s = np.log10(np.array(err_s))
                err_s = err_s[1:] - err_s[:-1]
                rate_s.append(np.median(err_s))

                # Compute error for G - Sinkhorn
                err_ti = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(5000):
                    f_tmp = f.copy()
                    f, g = numpy_loop(f, a, b, C, epst, rhot)
                    t = rescale_potentials(f, g, a, b, rhot)
                    err_ti.append(np.amax(np.abs(f + t - fr)))
                    if np.amax(np.abs(f + t - fr)) < 1e-12:
                        break
                err_ti = np.log10(np.array(err_ti))
                err_ti = err_ti[1:] - err_ti[:-1]
                rate_ti.append(np.median(err_ti))

                # Compute error for H - Sinkhorn
                err_f = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(5000):
                    f_tmp = f.copy()
                    f, g = faster_loop(f, a, b, C, epst, rhot)
                    t = rescale_potentials(f, g, a, b, rhot)
                    err_f.append(np.amax(np.abs(f + t - fr)))
                    if np.amax(np.abs(f + t - fr)) < 1e-12:
                        break
                err_f = np.log10(np.array(err_f))
                err_f = err_f[1:] - err_f[:-1]
                rate_f.append(np.median(err_f))

            # # Error for Sinkhorn
            # fb, gb = np.zeros_like(a), np.zeros_like(b)
            # for i in range(50000):
            #     f_tmp = fr.copy()
            #     fb, gb = balanced_loop(fb, a, b, C, epst)
            #     if hilbert_norm(fr - f_tmp) < 1e-15:
            #         break
            # err_b = []
            # f, g = np.zeros_like(a), np.zeros_like(b)
            # for i in range(5000):
            #     f_tmp = f.copy()
            #     f, g = balanced_loop(f, a, b, C, epst)
            #     err_b.append(hilbert_norm(f - fb))
            #     if hilbert_norm(f - fb) < 1e-12:
            #         break
            # err_b = np.log10(np.array(err_b))
            # err_b = err_b[1:] - err_b[:-1]
            # print(f"BALANCED RATE for {epst}", np.median(err_b))


            # Plot results
            np.save(
                path + "/ratekl/" + f"rate_f_sinkhorn_kl_eps{epst}.npy",
                np.array(rate_s))
            np.save(
                path + "/ratekl/" + f"rate_g_sinkhorn_kl_eps{epst}.npy",
                np.array(rate_ti))
            np.save(
                path + "/ratekl/" + f"rate_h_sinkhorn_kl_eps{epst}.npy",
                np.array(rate_f))


    ###########################################################################
    # Make plots
    ###########################################################################
    p = 0.97
    colors = ['', 'cornflowerblue', 'indianred']
    markers = ['x', 'o', 'v']
    markevery = 2
    f, ax = plt.subplots(1, 1, figsize=(p * 5, p * 4))

    rho_scale = 10 ** np.load(path + "/ratekl/" + f"rho_scale.npy")

    for r in range(1, len(eps_l)):
        epst = 10 ** eps_l[r]
        rate_s = np.load(
            path + "/ratekl/" + f"rate_f_sinkhorn_kl_eps{epst}.npy")
        rate_f = np.load(
            path + "/ratekl/" + f"rate_h_sinkhorn_kl_eps{epst}.npy")
        ax.plot(rho_scale, 10 ** rate_s, c=colors[r], linestyle='dashed',
                     label=f'$S,\,\epsilon=${epst}', marker=markers[0],
                     markevery=markevery)
        ax.plot(rho_scale, 10 ** rate_f, c=colors[r],
                     label=f'$TI,\,\epsilon=${epst}', marker=markers[2],
                     markevery=markevery)

    ax.legend(fontsize=11, ncol=2, columnspacing=0.5, handlelength=1.3,
                   loc=(.4, .02))

    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Marginal parameter $\\rho$', fontsize=15)
    ax.set_title('KL entropy', fontsize=18)
    ax.set_ylabel('Contraction rate', fontsize=15)

    plt.tight_layout()
    plt.savefig(path + "/paper/" + 'plot_log_contraction_rate_kl_fast.pdf')
    plt.show()