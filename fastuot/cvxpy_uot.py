import cvxpy as cp
import numpy as np


def primal_via_cvxpy(a, b, x, y, p, rho):
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


def dual_via_cvxpy(a, b, x, y, p, rho, cpsolv='ECOS', tol=1e-7, niter=5000):
    C = np.abs(x[:, None] - y[None, :]) ** p
    f = cp.Variable((x.shape[0]))
    g = cp.Variable((y.shape[0]))
    inta = cp.sum(cp.multiply(a, cp.exp(-f / rho)))
    intb = cp.sum(cp.multiply(b, cp.exp(-g / rho)))
    constr = [f[:, None] + g[None, :] <= C]
    objective = cp.Minimize(inta + intb)
    prob = cp.Problem(objective, constr)
    if cpsolv == 'SCS':
        result = prob.solve(max_iters=niter, verbose=False,
                            solver=cp.SCS, eps=tol)
    elif cpsolv == 'ECOS':
        result = prob.solve(max_iters=niter, verbose=False,
                            solver=cp.ECOS, abstol=tol)
    else:
        raise Exception("cpsolv should be string 'ECOS' or 'SCS'.")
    return result, constr, f, g