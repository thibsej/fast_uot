import os
import time

import numpy as np
import matplotlib.pyplot as plt

from fastuot.uot1d import solve_ot, rescale_potentials, invariant_dual_loss, \
    homogeneous_line_search
from fastuot.cvxpy_uot import dual_via_cvxpy

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/paper/"
if not os.path.isdir(path):
    os.mkdir(path)


rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)
lw =2


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
    np.random.seed(6)
    n, m = 50, 50
    # a, x, b, y = generate_measure(n, m)
    a, x, b, y = generate_random_measure(n, m)


    # params
    p = 1.5
    rho = .05
    niter = 2000
    C = np.abs(x[:, None] - y[None, :]) ** p

    result, constr, fr, gr = dual_via_cvxpy(a, b, x, y, p, rho, cpsolv='ECOS', tol=1e-10)
    fr, gr = fr.value, gr.value
    plt.imshow(np.log10(C - fr[:, None] - gr[None, :]))
    plt.title('CVXPY')
    plt.show()

    ###########################################################################
    # Vanilla FW
    ###########################################################################
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
        # norm_fw.append(np.log(hilbert_norm(f - fr)))

    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('FW')
    plt.show()

    ###########################################################################
    # Vanilla FW with dual line search
    ###########################################################################
    # f, g = np.zeros_like(a), np.zeros_like(b)
    # dual_lfw, norm_lfw = [], []
    # for k in range(niter):
    #     transl = rescale_potentials(f, g, a, b, rho, rho)
    #     f, g = f + transl, g - transl
    #     norm_lfw.append(np.log10(np.amax(np.abs(f - fr))))
    #     dual_lfw.append(invariant_dual_loss(f, g, a, b, rho))
    #     A = np.exp(-f / rho) * a
    #     B = np.exp(-g / rho) * b
    #
    #     # update
    #     I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
    #     gamma = newton_line_search(f, g, fs - f, gs - g, a, b, rho, rho,
    #                                     nits=5)
    #     f = f + gamma * (fs - f)
    #     g = g + gamma * (gs - g)
    #
    # plt.imshow(np.log(C - f[:, None] - g[None, :]))
    # plt.title('Linesearch-FW')
    # plt.show()

    ###########################################################################
    # Vanilla FW with homogeneous line search
    ###########################################################################
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
        gamma = homogeneous_line_search(f, g, fs - f, gs - g, a, b, rho, rho,
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

    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('Homogeneous-FW')
    plt.show()

    ###########################################################################
    # Pairwise FW
    ###########################################################################
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

    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('PFW')
    plt.show()

    ###########################################################################
    # Plot dual loss and norm
    ###########################################################################
    # plt.plot(dual_fw, label='FW')
    # # plt.plot(dual_lfw, label='LFW')
    # plt.plot(dual_hfw, label='HFW')
    # plt.plot(dual_pfw, label='PFW')
    # plt.legend()
    # plt.title('DUAL')
    # plt.show()

    # Plot results
    t_fw = np.median(np.array(time_fw))
    t_hfw = np.median(np.array(time_hfw))
    t_pfw = np.median(np.array(time_pfw))
    plt.figure(figsize=(5, 4))
    # plt.plot(norm_lfw, label='LFW')
    plt.plot(t_hfw * np.arange(len(norm_hfw)),  np.array(norm_hfw),
             label='$HFW$', c='r', linewidth=lw)
    plt.plot(t_pfw * np.arange(len(norm_pfw)),  np.array(norm_pfw),
             label='$PFW$', c='g', linewidth=lw)
    plt.plot(t_fw * np.arange(len(norm_fw)),  np.array(norm_fw),
             label='$FW$', c='b', linewidth=lw)
    plt.xlabel('$Time$', fontsize=16)
    plt.ylabel('$\log_{10}\|f_t - f^*\|_\infty$', fontsize=16)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path + f'plot_fw_comparison.pdf')
    plt.show()
