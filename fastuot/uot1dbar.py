import numpy as np
import numba
from numba import jit
from tqdm import tqdm

# Numba issues a deprecation warning for lists...
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@jit(nopython=True)
def solve_balanced_barycenter(a, x, lam):
    """Computes the 1D Optimal Transport barycenters between several histograms.

    _Important:  np.sum(a[k]) should be the same for all k (balanced OT problems)._

    Solve the problem
        min_beta sum_k lam[k] * OT( alpha_k, beta )
    where OT is the 1D optimal transport distance for the squared Euclidean constant
    |x-y|^2, alpha_k is the k^th input measure
    (with positions x[k] and weights a[k], see below). The solution beta has
    weights P and positions y (see below).

    The size of the output barycenter is
        m := (sum_k n[k]) - k + 1

    This is computed by solving a multi-marginal problem, and it also output
    the dual variable solution of
        sum_k lam[k] * <a[k],f[k]>  s.t.  sum_k f[k][ik] <= C[i1,\ldots,iK]
    where C is the multi-marginal cost
        C[i1,\ldots,iK] = sum_k lam[k] |x[k][ik] - bar|^2
    where
        bar = sum_k lam[k] * x[k][ik]

    Parameters
    ----------
    a: list of K vectors, a[k] is the histogram #k

    x: list of K vectors, x[k] are the positions of the histogram #k, it *must* be increasing
        and have same length as a[k]

    lam: weights for the barycenters, should be positive and sum to 1,
        lam[k] is the weight for the k^th histogram.

    Returns
    ----------
    I: matrix of size (m,K), each I[:,k] is a
        vector of length m of increasing integer in range(len(n[p])

    P: vector of length m, weights of the barycenter.

    f: list of K vector, f[k] is the k^th dual potential
        associated to a[k] (and has same size).

    cost: dual cost, equal to
        sum_k lam[k] * np.sum( a[k]*f[k] )

    """

    # number of histograms
    K = len(a)
    n = np.zeros(K, dtype=np.int32)
    for k in range(K):
        n[k] = len(a[k])

    # initialize
    m = np.sum(n - 1) + 1  # size of the barycenter

    P = np.zeros(m)
    f = []
    for k in range(K):
        f.append(np.zeros(n[k]))
    I = np.zeros((m, K), dtype=np.int32)

    # evaluate the multi-marginal cost
    def eval_cost(L):
        m = 0
        for k in range(K):  # barycenter point
            m += lam[k] * x[k][L[k]]
        c = 0
        for k in range(K):
            c += (x[k][L[k]] - m) ** 2
        return c

    # initialize the dual potentials so that
    #   sum_k f[0,k] = cost([0,...,0])
    f[0][0] = eval_cost(np.zeros(K, dtype=np.int32))

    # will be progressively destroyed
    a1 = []
    for k in range(K):
        a1.append(a[k].copy())

    h = np.zeros(K)  # tmp variable
    for j in range(m - 1):
        # find which input k0 to flush, take the one with minimum mass
        for k in range(K):
            h[k] = a1[k][I[j, k]]
            if I[j, k] == n[k] - 1:  # boundary reached, should not select this
                h[k] = 1e6
        k0 = np.argmin(h)
        i0 = I[j, k0]
        # nobody moves excepted k0
        for k in range(K):
            i = I[j, k]
            a1[k][i] = a1[k][i] - h[k0]
            if (k == k0):
                I[j + 1, k] = i + 1
            else:
                I[j + 1, k] = i
        P[j] = h[k0]  # transported mass

        # update dual potentials
        L = np.zeros(K, dtype=np.int32)
        ff = 0
        for k in range(K):
            if k != k0:
                i = I[j, k]
                L[k] = i
                ff = ff + f[k][i]
            else:
                L[k0] = i0 + 1
        f[k0][i0 + 1] = eval_cost(L) - ff
    # remaining mass, should be the same for all,
    # P[m-1] = a1[n[k]-1,k] must be constant
    P[m - 1] = a1[0][n[0] - 1]

    # position of the barycenters
    y = np.zeros(m)
    for j in range(m):
        for k in range(K):
            i = I[j, k]
            y[j] = y[j] + lam[k] * x[k][i]

    # dual cost sum_k lam[k]*<ak,fk>
    cost = 0
    for k in range(K):
        cost = cost + lam[k] * np.sum(a[k] * f[k])

    return I, P, y, f, cost


def solve_unbalanced_barycenter(a, x, lam, rho, niter=100):
    """Compute unbalanced OT barycenter using Frank-Wolfe's algorithm.

    Parameters
    ----------

    See solve(), with the extra parameters:

    rho: unbalanced parameter (increasing it to make the problem more balanced)

    niter: number of FW iterations.


    Returns
    ----------

    See solve() (except cost is not returned for now).

    """
    # number of histograms
    K = len(a)
    n = np.zeros(K, dtype=np.int32)
    for k in range(K):
        n[k] = len(a[k])
    # initialization
    f, a1 = [], []
    for k in range(K):
        f.append(np.zeros(n[k]))
        a1.append(np.zeros(n[k]))
    # F-W iterations
    cost = []
    for it in tqdm(range(niter)):
        # optimal translation
        A, c = np.zeros(K), np.zeros(K)
        for k in range(K):
            A[k] = np.sum(a[k] * np.exp(-f[k] / (rho * lam[k])))
        for k in range(K):
            c[k] = rho * lam[k] * (np.log(A[k]) - np.sum(lam * np.log(A)))
            f[k] += c[k]
        # modified histograms
        for k in range(K):
            a1[k] = a[k] * np.exp(-f[k] / (rho * lam[k]))
        # print(np.sum(a1[0]) , np.sum(a1[1]))
        # update dual potentials
        I, P, y, fs, _ = solve_balanced_barycenter(a1, x, lam)
        # F-W step
        gamma = 2 / (2 + it)  # fixed decaying weights
        for k in range(K):
            f[k] = (1 - gamma) * f[k] + gamma * fs[k]
        # Evaluate cost
        c, summass = 1., 0.
        for k in range(K):
            c = c * np.sum(a[k] * np.exp(-f[k] / rho))**(1 / K)
            summass = summass + rho * np.sum(a[k])
        cost.append(summass-c)
    return I, P, y, f, cost
