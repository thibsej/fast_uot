import os

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix

from fastuot.uot1d import solve_ot, rescale_potentials, solve_uot, \
    compute_shortest_path_support, build_potential_from_support, \
    dual_loss, invariant_dual_loss

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_comparison/"
if not os.path.isdir(path):
    os.mkdir(path)


def solver_via_cvxpy(a, b, x, y, p, rho):
    C = np.abs(x[:, None] - y[None, :]) ** p
    P = cp.Variable((x.shape[0], y.shape[0]))
    u = np.ones((x.shape[0], 1))
    v = np.ones((y.shape[0], 1))
    q = cp.sum(cp.kl_div(cp.matmul(P, v), a[:, None]))
    r = cp.sum(cp.kl_div(cp.matmul(P.T, u), b[:, None]))
    constr = [0 <= P]
    objective = cp.Minimize(cp.sum(cp.multiply(P, C)) + rho * q + rho * r)
    prob = cp.Problem(objective, constr)
    result = prob.solve()
    return result, constr, P


def dual_via_cvxpy(a, b, x, y, p, rho):
    C = np.abs(x[:, None] - y[None, :]) ** p
    f = cp.Variable((x.shape[0]))
    g = cp.Variable((y.shape[0]))
    inta = cp.sum(cp.multiply(a, cp.exp(-f / rho)))
    intb = cp.sum(cp.multiply(b, cp.exp(-g / rho)))
    constr = [f[:, None] + g[None, :] <= C]
    objective = cp.Minimize(inta + intb)
    prob = cp.Problem(objective, constr)
    result = prob.solve(abstol=1e-10)
    return result, constr, f, g


if __name__ == '__main__':
    # generate data
    n = int(15)
    m = int(16)
    np.random.seed(0)
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

    result, constr, fr, gr = dual_via_cvxpy(a, b, x, y, p, rho)
    fr, gr = fr.value, gr.value
    plt.imshow(np.log(C - fr[:, None] - gr[None, :]))
    plt.show()

    # Vanilla FW
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
    plt.show()

    # Pairwise FW
    f, g = np.zeros_like(a), np.zeros_like(b)
    dual_pfw, norm_pfw = [], []
    atoms = [[f, g]]
    weights = [1.]
    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        norm_pfw.append(np.log(np.amax(np.abs(f - fr))))  # TODO: L1 norm
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
        atoms.append([fs, gs])

        t = weights[itop]
        gamma = np.minimum(2. / (2. + k), t)
        l1 = invariant_dual_loss(f + gamma * (fs - fa), g + gamma * (gs - ga),
                                 a, b, rho)
        l2 = invariant_dual_loss(f + t * (fs - fa), g + t * (gs - ga), a, b,
                                 rho)
        if l1 < l2:
            f = f + t * (fs - fa)
            g = g + t * (gs - ga)
            weights.append(t)
            atoms.pop(itop)
            weights.pop(itop)
        else:
            f = f + gamma * (fs - fa)
            g = g + gamma * (gs - ga)
            weights.append(gamma)
            weights[itop] = weights[itop] - gamma
            if weights[itop] <= 0.:
                atoms.pop(itop)
                weights.pop(itop)
        print(f"Weights at time {k}", len(weights), len(set(weights)))
    plt.imshow(np.log(C - f[:, None] - g[None, :]))
    plt.show()

    # Plot results
    plt.plot(dual_fw, label='FW')
    plt.plot(dual_pfw, label='PFW')
    plt.legend()
    plt.title('DUAL')
    plt.show()

    # Plot results
    plt.plot(norm_fw, label='FW')
    plt.plot(norm_pfw, label='PFW')
    plt.legend()
    plt.title('NORM')
    plt.show()
