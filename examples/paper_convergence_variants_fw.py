import os
import time

import numpy as np
import matplotlib.pyplot as plt

from fastuot.uot1d import solve_ot, rescale_potentials, invariant_dual_loss, \
    homogeneous_line_search, solve_uot
from fastuot.cvxpy_uot import dual_via_cvxpy

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "/paper/"):
    os.mkdir(path + "/paper/")
if not os.path.isdir(path + "/variantfw/"):
    os.mkdir(path + "/variantfw/")


rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)



def normalize(x):
    return x / np.sum(x)


def gauss(grid, mu, sig):
    return np.exp(-0.5 * ((grid-mu) / sig) ** 2)


def hilbert_norm(f):
    return np.amax(np.abs(f)) - np.amin(np.abs(f))


def generate_random_measure(n, m):
    a = normalize(np.random.uniform(size=n))
    b = normalize(np.random.uniform(size=m))
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=m))
    return a, x, b, y


def generate_measure(n, m):
    x = np.linspace(0.2, 0.4, num=n)
    a = np.zeros_like(x)
    a[:n // 2] = 2.
    a[n // 2:] = 3.
    y = np.linspace(0.45, 0.95, num=m)
    a = normalize(a)
    b = normalize(gauss(y, 0.6, 0.03)
                  + gauss(y, 0.7, 0.03)
                  + gauss(y, 0.8, 0.03))
    return a, x, b, y


if __name__ == '__main__':
    compute_data = False
    np.random.seed(6)
    n, m = 5000, 5001
    a, x, b, y = generate_random_measure(n, m)
    # a, x, b, y = generate_measure(n, m)

    # params
    p = 1.5
    rho = .1
    niter = 10000

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        # result, constr, fr, gr = dual_via_cvxpy(a, b, x, y, p, rho,
        #                                         cpsolv='SCS',
        #                                         tol=1e-10)
        # fr, gr = fr.value, gr.value
        _, _, _, fr, _, _ = solve_uot(a, b, x, y, p, rho, niter=200000)
        np.save(path + "/variantfw/" + "ref_pot_cvxpy.npy", fr)

        #######################################################################
        # Vanilla FW
        #######################################################################
        f, g = np.zeros_like(a), np.zeros_like(b)
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        dual_fw, norm_fw = [], []
        time_fw = []
        for k in range(niter):
            t0 = time.time()
            A = np.exp(-f / rho) * a
            B = np.exp(-g / rho) * b

            # update
            I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
            gamma = 2. / (2. + k)  # fixed decaying weights
            f = f + gamma * (fs - f)
            g = g + gamma * (gs - g)
            transl = rescale_potentials(f, g, a, b, rho, rho)
            f, g = f + transl, g - transl
            if np.isnan(f).any():
                break

            time_fw.append(time.time() - t0)
            norm_fw.append(np.log10(np.amax(np.abs(f - fr))))
            dual_fw.append(invariant_dual_loss(f, g, a, b, rho))

        #######################################################################
        # Vanilla FW with homogeneous line search
        #######################################################################
        f, g = np.zeros_like(a), np.zeros_like(b)
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        dual_hfw, norm_hfw = [], []
        time_hfw = []
        for k in range(niter):
            t0 = time.time()
            A = np.exp(-f / rho) * a
            B = np.exp(-g / rho) * b

            # update
            I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
            gamma = homogeneous_line_search(f, g, fs - f, gs - g, a, b, rho,
                                            rho,
                                            nits=5)
            f = f + gamma * (fs - f)
            g = g + gamma * (gs - g)
            transl = rescale_potentials(f, g, a, b, rho, rho)
            f, g = f + transl, g - transl
            if np.isnan(f).any():
                break

            time_hfw.append(time.time() - t0)
            norm_hfw.append(np.log10(np.amax(np.abs(f - fr))))
            # norm_hfw.append(np.log(hilbert_norm(f - fr)))
            dual_hfw.append(invariant_dual_loss(f, g, a, b, rho))

        #######################################################################
        # Pairwise FW
        #######################################################################
        f, g = np.zeros_like(a), np.zeros_like(b)
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        dual_pfw, norm_pfw = [], []
        time_pfw = []
        atoms = [[f, g]]
        weights = [1.]
        for k in range(niter):
            t0 = time.time()
            A = np.exp(-f / rho) * a
            B = np.exp(-g / rho) * b

            # update
            I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)

            # Find best ascent direction
            score = np.inf
            itop = 0
            for i in range(len(atoms)):
                [ft, gt] = atoms[i]
                dscore = np.sum(A * ft) + np.sum(B * gt)
                if dscore < score:
                    itop = i
                    score = dscore
                    fa, ga = ft, gt

            # Check existence of atom in dictionary
            jtop = -1
            for i in range(len(atoms)):
                [ft, gt] = atoms[i]
                if np.array_equal(ft, fs) and np.array_equal(gt, gs):
                    jtop = i
                    break
            # print("if index in dictionary", jtop)
            if jtop == -1:
                atoms.append([fs, gs])
                weights.append(0.)

            gamma = homogeneous_line_search(f, g, fs-fa, gs-ga, a, b, rho, rho,
                                            nits=5, tmax=weights[itop])
            f = f + gamma * (fs - fa)
            g = g + gamma * (gs - ga)
            transl = rescale_potentials(f, g, a, b, rho, rho)
            f, g = f + transl, g - transl

            weights[jtop] = weights[jtop] + gamma
            weights[itop] = weights[itop] - gamma
            if weights[itop] <= 0.:
                atoms.pop(itop)
                weights.pop(itop)
            if np.isnan(f).any():
                break

            time_pfw.append(time.time() - t0)
            norm_pfw.append(np.log10(np.amax(np.abs(f - fr))))
            # norm_pfw.append(np.log(hilbert_norm(f - fr)))
            dual_pfw.append(invariant_dual_loss(f, g, a, b, rho))

        # save data
        t_fw = np.median(np.array(time_fw))
        t_hfw = np.median(np.array(time_hfw))
        t_pfw = np.median(np.array(time_pfw))
        np.save(path + "/variantfw/" + "time_comput_fw.npy",
                np.array([t_fw, t_hfw, t_pfw]))
        np.save(path + "/variantfw/" + "err_fw.npy",
                np.array(norm_fw))
        np.save(path + "/variantfw/" + "err_hfw.npy",
                np.array(norm_hfw))
        np.save(path + "/variantfw/" + "err_pfw.npy",
                np.array(norm_pfw))

    lw = 1.5
    colors = ['cornflowerblue', 'indianred', 'mediumseagreen']

    # Plot results
    time_arr = np.load(path + "/variantfw/" + "time_comput_fw.npy")
    t_fw, t_hfw, t_pfw = time_arr[0], time_arr[1], time_arr[2]
    err_fw = np.load(path + "/variantfw/" + "err_fw.npy")
    err_hfw = np.load(path + "/variantfw/" + "err_hfw.npy")
    err_pfw = np.load(path + "/variantfw/" + "err_pfw.npy")
    plt.figure(figsize=(4, 2.5))
    plt.plot(t_pfw * np.arange(1., len(err_pfw) + 1),  10**np.array(err_pfw),
             label='PFW', c=colors[2], linewidth=lw)
    plt.plot(t_hfw * np.arange(1., len(err_hfw) + 1),  10**np.array(err_hfw),
             label='HFW', c=colors[1], linewidth=lw)
    plt.plot(t_fw * np.arange(1., len(err_fw) + 1),  10**np.array(err_fw),
             label='FW', c=colors[0], linewidth=lw)
    plt.xlabel('Time', fontsize=15)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$\|f_t - f^*\|_\infty$', fontsize=15)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path + "/paper/" + f'plot_fw_comparison.pdf')
    plt.show()
