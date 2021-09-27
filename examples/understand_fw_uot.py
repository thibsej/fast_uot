import os

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix

from fastuot.uot1d import solve_ot, rescale_potentials, solve_uot, \
    compute_shortest_path_support, build_potential_from_support, dual_loss

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_comparison/"
if not os.path.isdir(path):
    os.mkdir(path)
# else:
#     for f in path:
#         os.remove(f)


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
    constr = [f[:,None] + g[None,:] <= C]
    objective = cp.Minimize(inta + intb)
    prob = cp.Problem(objective, constr)
    result = prob.solve(abstol=1e-10)
    return result, constr, f, g


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
    n = int(150)
    m = int(160)
    # np.random.seed(0)
    normalize = lambda p: p / np.sum(p)
    a = normalize(np.random.uniform(size=n))
    b = normalize(np.random.uniform(size=m))
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=m))

    # params
    p = 1.5
    rho = .1
    niter = 100
    C = np.abs(x[:,None] - y[None,:]) **p

    # result, constr, P = solver_via_cvxpy(a, b, x, y, p, rho)
    result, constr, f, g = dual_via_cvxpy(a, b, x, y, p, rho)
    print("DUAL CVXPY", f.value)
    # print("DUAL CVXPY", g.value)
    plt.imshow(np.log(constr[0].dual_value + 1e-10))
    # plt.imshow(np.log(P.value + 1e-10))
    plt.title('CVXPY')
    plt.savefig(path + 'img_ref_cvxpy.png')
    plt.clf()

    # I, J, P, f, g, cost = solve_uot(a, b, x, y, p, rho, rho, niter=niter,
    #                                 line_search='homogeneous')
    # pi = csr_matrix((P, (I, J)),
    #                 shape=(x.shape[0], y.shape[0])).toarray()
    #
    # plt.imshow(np.log(pi + 1e-10))
    # plt.title(f'Homogeneous')
    # plt.savefig(path + 'img_homogeneous.png')
    # plt.clf()
    # print("\nHomogeneous")
    # print("first potential", f)
    # print("Second potential", g)
    # # print("Constraint", C - (f[:,None] + g[None,:]))


    # I, J, P, f, g, cost = solve_uot(a, b, x, y, p, rho, rho, niter=niter,
    #                                 line_search='newton')
    # pi = csr_matrix((P, (I, J)),
    #                 shape=(x.shape[0], y.shape[0])).toarray()
    # plt.imshow(np.log(pi + 1e-10))
    # plt.title(f'newton')
    # plt.savefig(path + 'img_newton.png')
    # plt.clf()
    # print("\nNewton")
    # print("first potential", f)
    # print("Second potential", g)
    # # print("Constraint", C - (f[:,None] + g[None,:]))

    I, J, P, f, g, cost = solve_uot(a, b, x, y, p, rho, rho, niter=niter,
                                    line_search='default')
    pi = csr_matrix((P, (I, J)),
                    shape=(x.shape[0], y.shape[0])).toarray()
    gap = C - f[:,None] - g[None,:]
    plt.imshow(np.log(gap + 1e-15))
    # plt.imshow(np.log(pi + 1e-10))
    plt.title(f'default')
    plt.savefig(path + 'img_default.png')
    plt.clf()
    print("\nDefault")
    print("first potential", f)
    # print("Second potential", g)
    # print("Constraint", C - (f[:,None] + g[None,:]))

