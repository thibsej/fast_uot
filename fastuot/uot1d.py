import numpy as np
import numba
from numba import jit
import scipy as sp  # for sparse matrices
import scipy.sparse.linalg as sln
from scipy.sparse import csr_matrix


@jit(nopython=True)
def solve_ot(a, b, x, y, p):
    """Computes the 1D Optimal Transport between two histograms.

    _Important: one should have np.sum(a)=np.sum(b)._

    _Important:_ x and y needs to be sorted.

    Parameters
    ----------
    a: vector of length n with positive entries

    b: vector of length m with positive entries

    x: vector of real of length n

    y: vector of real of length m

    p: real, should >= 1


    Returns
    ----------
    I: vector of length q=n+m-1 of increasing integer in {0,...,n-1}

    J: vector of length q of increasing integer in {0,...,m-1}

    P: vector of length q of positive values of length q

    f: dual vector of length n

    g: dual vector of length m

    cost: (dual) OT cost
        sum a_i f_i + sum_j b_j f_j
        It should be equal to the primal cost
        = sum_k |x(i)-y(j)|^p where i=I(k), j=J(k)
    """
    n = len(a)
    m = len(b)
    q = m + n - 1
    a1 = a.copy()
    b1 = b.copy()
    I = np.zeros(q).astype(numba.int64)
    J = np.zeros(q).astype(numba.int64)
    P = np.zeros(q)
    f = np.zeros(n)
    g = np.zeros(m)
    g[0] = np.abs(x[0] - y[0]) ** p
    for k in range(q - 1):
        i = I[k]
        j = J[k]
        if (a1[i] < b1[j]) and (i < n - 1):
            I[k + 1] = i + 1
            J[k + 1] = j
            f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
        elif (a1[i] > b1[j]) and (j < m - 1):
            I[k + 1] = i
            J[k + 1] = j + 1
            g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
        elif i == n - 1:
            I[k + 1] = i
            J[k + 1] = j + 1
            g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
        elif j == m - 1:
            I[k + 1] = i + 1
            J[k + 1] = j
            f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
        t = min(a1[i], b1[j])
        P[k] = t
        a1[i] = a1[i] - t
        b1[j] = b1[j] - t
    P[k + 1] = max(a1[-1], b1[-1])  # remaining mass
    cost = np.sum(f * a) + np.sum(g * b)
    return I, J, P, f, g, cost


def logsumexp(f, a, stable_lse=True):
    """
    Computes the logsumexp operation, in stable form or not

    Parameters
    ----------
    f: numpy array of size n
    dual potential

    a: numpy array of size n with positive entries
    weights of measure

    stable_lse: bool
    If true, computes the logsumexp in stable form

    Return
    ------
    Float
    """
    if not stable_lse:
        return np.log(np.sum(a * np.exp(f)))
    else:
        xm = np.amax(f + np.log(a))
        return xm + np.log(np.sum(np.exp(f + np.log(a) - xm)))


def rescale_potentials(f, g, a, b, rho1, rho2, stable_lse=True):
    tau = (rho1 * rho2) / (rho1 + rho2)
    transl = tau * (logsumexp(-f / rho1, a, stable_lse=stable_lse) -
                    logsumexp(-g / rho2, b, stable_lse=stable_lse))
    return transl


def rescale_measure(a, b):
    z = np.sqrt(np.sum(a) / np.sum(b))
    return a / z, b * z


def dual_loss(f, g, a, b, rho1, rho2=None):
    if rho2 is None:
        rho2 = rho1
    loss = rho1 * np.sum(a * (1 - np.exp(-f / rho1))) \
           + rho2 * np.sum(b * (1 - np.exp(-g / rho2)))
    return loss


def invariant_dual_loss(f, g, a, b, rho1, rho2=None):
    if rho2 is None:
        rho2 = rho1
    loss = rho1 * np.sum(a) + rho2 * np.sum(b)
    tau1, tau2 = 1. / (1. + (rho2 / rho1)), 1. / (1. + (rho1 / rho2))
    int_a = np.sum(a * np.exp(-f / rho1))
    int_b = np.sum(b * np.exp(-g / rho2))
    loss = loss - (rho1 + rho2) * (int_a ** tau1) * (int_b ** tau2)
    return loss


def primal_dual_gap(a, b, x, y, p, f, g, P, I, J, rho1, rho2=None):
    if rho2 is None:
        rho2 = rho1
    prim = np.sum(P * np.abs(x[I] - y[J])**p)
    dual = np.sum(f * np.exp(-f / rho1) * a) \
           + np.sum(g * np.exp(-g / rho2) * b)
    return prim - dual


def solve_uot(a, b, x, y, p, rho1, rho2=None, niter=100, tol=1e-10,
              greed_init=True, line_search='default', stable_lse=True):
    assert line_search in ['homogeneous', 'newton', 'default']
    if rho2 is None:
        rho2 = rho1

    # Initialize potentials
    if greed_init:
        f, g = init_greed_uot(a, b, x, y, p, rho1, rho2)
    else:
        f, g = np.zeros_like(a), np.zeros_like(b)

    for k in range(niter):
        # Output FW descent direction
        transl = rescale_potentials(f, g, a, b, rho1, rho2,
                                    stable_lse=stable_lse)
        f, g = f + transl, g - transl
        A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
        I, J, P, fd, gd, cost = solve_ot(A, B, x, y, p)

        # Line search - convex update
        if line_search == 'homogeneous':
            t = homogeneous_line_search(f, g, fd - f, gd - g,
                                        a, b, rho1, rho2, nits=5)
        if line_search == 'newton':
            t = newton_line_search(f, g, fd - f, gd - g,
                                   a, b, rho1, rho2, nits=5)
        if line_search == 'default':
            t = 2. / (2. + k)
        f = f + t * (fd - f)
        g = g + t * (gd - g)

        pdg = primal_dual_gap(a, b, x, y, p, f, g, P, I, J, rho1, rho2=None)
        if pdg < tol:
            break

    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2,
                                stable_lse=stable_lse)
    f, g = f + transl, g - transl
    A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
    I, J, P, _, _, cost = solve_ot(A, B, x, y, p)
    return I, J, P, f, g, cost


def pairwise_solve_uot(a, b, x, y, p, rho1, rho2=None, niter=100, tol=1e-10,
                       greed_init=True, stable_lse=True):
    if rho2 is None:
        rho2 = rho1

    # Initialize potentials
    if greed_init:
        f, g = init_greed_uot(a, b, x, y, p, rho1, rho2)
    else:
        f, g = np.zeros_like(a), np.zeros_like(b)

    # Store atoms for pairwise steps
    atoms = [[f, g]]
    weights = [1.]

    for k in range(niter):
        transl = rescale_potentials(f, g, a, b, rho1, rho2)
        f, g = f + transl, g - transl
        A = np.exp(-f / rho1) * a
        B = np.exp(-g / rho2) * b

        # update
        I, J, P, fd, gd, _ = solve_ot(A, B, x, y, p)

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
            if np.array_equal(ft, fd) and np.array_equal(gt, gd):
                jtop = i
                break
        if jtop == -1:
            atoms.append([fd, gd])
            weights.append(0.)

        gamma = homogeneous_line_search(f, g, fd - fa, gd - ga,
                                        a, b, rho1, rho2,
                                        nits=5, tmax=weights[itop])
        f = f + gamma * (fd - fa)
        g = g + gamma * (gd - ga)
        weights[jtop] = weights[jtop] + gamma
        weights[itop] = weights[itop] - gamma
        if weights[itop] <= 0.:
            atoms.pop(itop)
            weights.pop(itop)

        pdg = primal_dual_gap(a, b, x, y, p, f, g, P, I, J, rho1, rho2=None)
        if pdg < tol:
            break

    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2, stable_lse=stable_lse)
    f, g = f + transl, g - transl
    A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
    I, J, P, _, _, cost = solve_ot(A, B, x, y, p)
    return I, J, P, f, g, cost


def homogeneous_line_search(fin, gin, d_f, d_g, a, b, rho1, rho2, nits,
                            tmax=1.):
    """
    Convex interpolation ft = (1 - t) * fin + t * fout.

    Parameters
    ----------
    fin
    gin
    d_f
    d_g
    rho1
    rho2
    nits
    tmax

    Returns
    -------
    """
    t = 0.5
    tau1, tau2 = 1. / (1. + (rho2 / rho1)), 1. / (1. + (rho1 / rho2))
    for k in range(nits):
        ft = fin + t * d_f
        gt = gin + t * d_g

        # Compute derivatives for F
        a_z = np.sum(a * np.exp(-ft / rho1))
        a_p = - np.sum(a * np.exp(-ft / rho1) * (-d_f)) / rho1
        a_s = np.sum(a * np.exp(-ft / rho1) * (-d_f) ** 2) / rho1 ** 2
        a_s = tau1 * a_s * a_z ** (tau1 - 1) \
              + tau1 * (tau1 - 1) * a_p ** 2 * a_z ** (tau1 - 2)
        a_p = tau1 * a_p * a_z ** (tau1 - 1)
        a_z = a_z ** tau1

        # Compute derivatives for G
        b_z = np.sum(b * np.exp(-gt / rho2))
        b_p = - np.sum(b * np.exp(-gt / rho2) * (-d_g)) / rho2
        b_s = np.sum(b * np.exp(-gt / rho2) * (-d_g) ** 2) / rho2 ** 2
        b_s = tau2 * b_s * b_z ** (tau2 - 1) \
              + tau2 * (tau2 - 1) * b_p ** 2 * b_z ** (tau2 - 2)
        b_p = tau2 * b_p * b_z ** (tau2 - 1)
        b_z = b_z ** tau2

        # Compute damped Newton step
        loss_p = a_p * b_z + a_z * b_p
        loss_s = a_s * b_z + 2 * a_p * b_p + a_z * b_s
        t = t + (loss_p / loss_s) / (1 + np.sqrt(loss_p ** 2 / loss_s))

        # Clamp to keep a convex combination
        t = np.maximum(np.minimum(t, tmax), 0.)
    return t


def newton_line_search(fin, gin, d_f, d_g, a, b, rho1, rho2, nits, tmax=1.):
    t = 0.5
    for k in range(nits):
        ft = fin + t * d_f
        gt = gin + t * d_g

        grad = np.sum(a * d_f * np.exp(-ft / rho1)) \
               + np.sum(b * d_g * np.exp(-gt / rho2))
        hess = -np.sum(
            a * d_f ** 2 * np.exp(-ft / rho1)) / rho1 \
               - np.sum(
            b * d_g ** 2 * np.exp(-(gt) / rho2)) / rho2
        t = t - (grad / hess)
        t = np.maximum(np.minimum(t, tmax), 0.)
    return t


###############################################################################
# GREEDY INITIALIZATION
###############################################################################

def init_greed_uot(a, b, x, y, p, rho1, rho2=None):
    if rho2 is None:
        rho2 = rho1

    _, _, _, fb, gb, _ = solve_ot(a / np.sum(a), b / np.sum(b), x, y, p)
    fc, gc = lazy_potential(x, y, p)

    # Output best convex combination
    t = homogeneous_line_search(fb, gb, fc - fb, gc - gb, a, b, rho1, rho2,
                                nits=3)
    ft = (1 - t) * fb + t * fc
    gt = (1 - t) * gb + t * gc
    return ft, gt


@jit(nopython=True)
def lazy_potential2(x, y, p):
    """Computes the 1D Optimal Transport between two histograms.

    _Important: one should have np.sum(a)=np.sum(b)._

    _Important:_ x and y needs to be sorted.

    Parameters
    ----------
    x: vector of real of length n

    y: vector of real of length m

    p: real, should >= 1


    Returns
    ----------
    f: dual vector of length n

    g: dual vector of length m
    """
    n = x.shape[0]
    m = y.shape[0]
    q = m + n - 1
    i = 0
    j = 0
    f = np.zeros(n)
    g = np.zeros(m)
    g[0] = np.abs(x[0] - y[0]) ** p
    for k in range(q - 1):
        if i == n - 1:
            j += 1
            c12 = np.abs(x[i] - y[j]) ** p
            g[j] = c12 - f[i]
            continue
        elif j == m - 1:
            i += 1
            c21 = np.abs(x[i] - y[j]) ** p
            f[i] = c21 - g[j]
            continue

        c12 = np.abs(x[i] - y[j + 1]) ** p
        c21 = np.abs(x[i + 1] - y[j]) ** p
        if c12 > c21:
            i += 1
            f[i] = c21 - g[j]
        elif c12 < c21:
            j += 1
            g[j] = c12 - f[i]
    return f, g

@jit(nopython=True)
def lazy_potential(x, y, p, diagonal=True):
    """Computes the 1D Optimal Transport between two histograms.

    _Important: one should have np.sum(a)=np.sum(b)._

    _Important:_ x and y needs to be sorted.

    Parameters
    ----------
    x: vector of real of length n

    y: vector of real of length m

    p: real, should >= 1


    Returns
    ----------
    f: dual vector of length n

    g: dual vector of length m
    """
    n = x.shape[0]
    m = y.shape[0]
    i = 0
    j = 0
    f = np.zeros(n)
    g = np.zeros(m)
    g[0] = np.abs(x[0] - y[0]) ** p
    while (i < n - 1) or (j < m - 1):
        if i == n - 1:
            j += 1
            c12 = np.abs(x[i] - y[j]) ** p
            g[j] = c12 - f[i]
            continue
        elif j == m - 1:
            i += 1
            c21 = np.abs(x[i] - y[j]) ** p
            f[i] = c21 - g[j]
            continue

        c12 = np.abs(x[i] - y[j + 1]) ** p
        c21 = np.abs(x[i + 1] - y[j]) ** p
        if diagonal:
            c22 = np.abs(x[i + 1] - y[j + 1]) ** p
        if diagonal and (c22 < c12) and (c22 < c21):
            i += 1
            j += 1
            f[i] = 0.5 * (c22 + c21 - c12 + f[i-1] - g[j-1])
            g[j] = 0.5 * (c22 + c12 - c21 - f[i-1] + g[j-1])
        elif c12 > c21:
            i += 1
            f[i] = c21 - g[j]
        elif c12 < c21:
            j += 1
            g[j] = c12 - f[i]
    return f, g
