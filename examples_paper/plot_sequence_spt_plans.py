import os

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix

from fastuot.uot1d import solve_ot, rescale_potentials, dual_loss

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_support/"
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


def shortest_cost_path(c):
    I, J = [], []
    I.append(0)
    J.append(0)
    i, j = 0, 0
    while (i < c.shape[0]-1) and (j < c.shape[1]-1):
        c12 = c[i, (j + 1)]
        c21 = c[(i + 1), j]
        c22 = c[(i + 1), (j + 1)]
        if (c22 < c12) and (c22 < c21):
            i = i+1
            j = j+1
            I.append(i)
            J.append(j)
        elif c12 < c21:
            j = j+1
            I.append(i)
            J.append(j)
        else:
            i = i+1
            I.append(i)
            J.append(j)
    return I, J


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
    rho = .001
    niter = 300

    result, constr, P = solver_via_cvxpy(a, b, x, y, p, rho)
    plt.imshow(np.log(P.value + 1e-10))
    plt.title('CVXPY')
    plt.savefig(path + 'img_ref_cvxpy.png')
    plt.clf()

    c = np.abs(x[:, None] - y[None, :]) ** p
    Ic, Jc = shortest_cost_path(c)
    plt.scatter(Jc, Ic, color='r')
    plt.imshow(np.abs(x[:,None]-y[None,:])**p)
    plt.title('COST')
    plt.savefig(path + 'img_cost.png')
    plt.clf()

    # start iterations
    dual_value, primal_value = [], []
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        A = np.exp(-f / rho) * a
        B = np.exp(-g / rho) * b
        dual_value.append(dual_loss(f, g, a, b, rho, rho2=rho))

        # update
        I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
        gamma = 2. / (2. + k)  # fixed decaying weights
        f = (1 - gamma) * f + gamma * fs
        g = (1 - gamma) * g + gamma * gs

        pi = csr_matrix((P, (I, J)),
                        shape=(x.shape[0], y.shape[0])).toarray()
        plt.imshow(np.log(pi + 1e-10))
        plt.title(f'Iter {k}')
        plt.savefig(path + f'img_fw_iter{k}.png')
        plt.clf()

    plt.plot(dual_value[10:], label='dual')
    plt.title('Dual Value')
    plt.legend()
    plt.savefig(path + f'img_dual_cost.png')
    plt.clf()

    print("first potential", f)
    print("Second potential", g)