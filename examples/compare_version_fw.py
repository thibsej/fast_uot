import os

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix

from fastuot.uot1d import solve_ot, rescale_potentials, solve_uot, \
    dual_loss, invariant_dual_loss, homogeneous_line_search, \
    newton_line_search, primal_dual_gap, lazy_potential
from fastuot.cvxpy_uot import dual_via_cvxpy

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_comparison/"
if not os.path.isdir(path):
    os.mkdir(path)


if __name__ == '__main__':
    # generate data
    n = int(15)
    m = int(16)
    # np.random.seed(0)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    b = normalize(np.random.uniform(size=m))
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=m))

    # params
    p = 1.5
    rho = .1
    niter = 500
    C = np.abs(x[:, None] - y[None, :]) ** p

    result, constr, fr, gr = dual_via_cvxpy(a, b, x, y, p, rho, tol=1e-10)
    fr, gr = fr.value, gr.value
    plt.imshow(np.log(C - fr[:, None] - gr[None, :]))
    plt.title('CVXPY')
    plt.show()

    ###########################################################################
    # Vanilla FW
    ###########################################################################
    f, g = np.zeros_like(a), np.zeros_like(b)
    dual_fw, norm_fw = [], []
    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        norm_fw.append(np.log(np.amax(np.abs(f - fr))))
        dual_fw.append(invariant_dual_loss(f, g, a, b, rho))
        A = np.exp(-f / rho) * a
        B = np.exp(-g / rho) * b

        # update
        I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
        gamma = 2. / (2. + k)  # fixed decaying weights
        f = f + gamma * (fs - f)
        g = g + gamma * (gs - g)

    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('FW')
    plt.show()

    ###########################################################################
    # Vanilla FW with dual line search
    ###########################################################################
    f, g = np.zeros_like(a), np.zeros_like(b)
    dual_lfw, norm_lfw = [], []
    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        norm_lfw.append(np.log(np.amax(np.abs(f - fr))))
        dual_lfw.append(invariant_dual_loss(f, g, a, b, rho))
        A = np.exp(-f / rho) * a
        B = np.exp(-g / rho) * b

        # update
        I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
        gamma = newton_line_search(f, g, fs - f, gs - g, a, b, rho, rho,
                                        nits=5)
        f = f + gamma * (fs - f)
        g = g + gamma * (gs - g)

    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('Linesearch-FW')
    plt.show()

    ###########################################################################
    # Vanilla FW with homogeneous line search
    ###########################################################################
    f, g = np.zeros_like(a), np.zeros_like(b)
    dual_hfw, norm_hfw = [], []
    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        norm_hfw.append(np.log(np.amax(np.abs(f - fr))))
        dual_hfw.append(invariant_dual_loss(f, g, a, b, rho))
        A = np.exp(-f / rho) * a
        B = np.exp(-g / rho) * b

        # update
        I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
        gamma = homogeneous_line_search(f, g, fs - f, gs - g, a, b, rho, rho,
                                        nits=5)
        f = f + gamma * (fs - f)
        g = g + gamma * (gs - g)

    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('Homogeneous-FW')
    plt.show()

    ###########################################################################
    # Pairwise FW
    ###########################################################################
    f, g = np.zeros_like(a), np.zeros_like(b)
    dual_pfw, norm_pfw = [], []
    atoms = [[f, g]]
    weights = [1.]
    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        norm_pfw.append(np.log(np.amax(np.abs(f - fr))))
        dual_pfw.append(invariant_dual_loss(f, g, a, b, rho))
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
        # gamma = newton_line_search(f, g, fs - fa, gs - ga, a, b, rho, rho,
        #                            nits=5, tmax=weights[itop])
        f = f + gamma * (fs - fa)
        g = g + gamma * (gs - ga)
        weights[jtop] = weights[jtop] + gamma
        weights[itop] = weights[itop] - gamma
        if weights[itop] <= 0.:
            atoms.pop(itop)
            weights.pop(itop)
    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.title('PFW')
    plt.show()

    ###########################################################################
    # Plot dual loss and norm
    ###########################################################################
    plt.plot(dual_fw, label='FW')
    plt.plot(dual_lfw, label='LFW')
    plt.plot(dual_hfw, label='HFW')
    plt.plot(dual_pfw, label='PFW')
    plt.legend()
    plt.title('DUAL')
    plt.show()

    # Plot results
    plt.plot(norm_fw, label='FW')
    plt.plot(norm_lfw, label='LFW')
    plt.plot(norm_hfw, label='HFW')
    plt.plot(norm_pfw, label='PFW')
    plt.legend()
    plt.title('NORM')
    plt.show()
