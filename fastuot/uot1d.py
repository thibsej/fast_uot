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
    loss = rho1 * np.sum(a) + rho2 * np.sum(b)
    tau1, tau2 = 1. / (1. + (rho2 / rho1)), 1. / (1. + (rho1 / rho2))
    int_a = np.sum(a * np.exp(-f / rho1))
    int_b = np.sum(b * np.exp(-g / rho2))
    loss = loss - (rho1 + rho2) * (int_a ** tau1) * (int_b ** tau2)
    return loss


def solve_uot(a, b, x, y, p, rho1, rho2=None, niter=100, tol=1e-6,
              greed_init=True, line_search='default', stable_lse=True):
    assert line_search in ['homogeneous', 'newton', 'default']
    if rho2 is None:
        rho2 = rho1

    # Initialize potentials
    if greed_init:
        f, g = init_greed_uot(a, b, x, y, p, rho1, rho2)
    else:
        f, g = np.zeros_like(a), np.zeros_like(b)

    # Output FW descent direction
    transl = rescale_potentials(f, g, a, b, rho1, rho2, stable_lse=stable_lse)
    f, g = f + transl, g - transl
    A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
    I, J, P, fb, gb, cost = solve_ot(A, B, x, y, p)
    for k in range(niter):
        f_tmp = f

        # Line search - convex update
        if line_search == 'homogeneous':  # TODO: To debug
            t = homogeneous_line_search(f, g, fb, gb, a, b, rho1, rho2, nits=5)
        if line_search == 'newton':  # TODO: To debug
            t = newton_line_search(f, g, fb, gb, a, b, rho1, rho2, nits=5)
        if line_search == 'default':
            t = 2. / (2. + k)
        f = (1 - t) * f + t * fb
        g = (1 - t) * g + t * gb

        # Compute next FW direction
        transl = rescale_potentials(f, g, a, b, rho1, rho2,
                                    stable_lse=stable_lse)
        f, g = f + transl, g - transl
        A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
        I, J, P, fb, gb, cost = solve_ot(A, B, x, y, p)
        if np.amax(np.abs(f - f_tmp)) < tol:
            break
    return I, J, P, f, g, cost


def homogeneous_line_search(fin, gin, fout, gout, a, b, rho1, rho2, nits):
    """
    Convex interpolation ft = (1 - t) * fin + t * fout.

    Parameters
    ----------
    fin
    gin
    fout
    gout
    rho1
    rho2
    nits

    Returns
    -------

    """
    t = 0.5
    tau1, tau2 = 1. / (1. + (rho2 / rho1)), 1. / (1. + (rho1 / rho2))
    for k in range(nits):
        ft = (1 - t) * fin + t * fout
        gt = (1 - t) * gin + t * gout

        # Compute derivatives for F
        a_z = np.sum(a * np.exp(-ft / rho1))
        a_p = - np.sum(a * np.exp(-ft / rho1) * (fin - fout)) / rho1
        a_s = np.sum(a * np.exp(-ft / rho1) * (fin - fout) ** 2) / rho1 ** 2
        a_s = tau1 * a_s * a_z ** (tau1 - 1) \
              + tau1 * (tau1 - 1) * a_p ** 2 * a_z ** (tau1 - 2)
        a_p = tau1 * a_p * a_z ** (tau1 - 1)
        a_z = a_z ** tau1

        # Compute derivatives for G
        b_z = np.sum(b * np.exp(-gt / rho2))
        b_p = - np.sum(b * np.exp(-gt / rho2) * (gin - gout)) / rho2
        b_s = np.sum(b * np.exp(-gt / rho2) * (gin - gout) ** 2) / rho2 ** 2
        b_s = tau2 * b_s * b_z ** (tau2 - 1) \
              + tau2 * (tau2 - 1) * b_p ** 2 * b_z ** (tau2 - 2)
        b_p = tau2 * b_p * b_z ** (tau2 - 1)
        b_z = b_z ** tau2

        # Compute damped Newton step
        loss_p = a_p * b_z + a_z * b_p
        loss_s = a_s * b_z + 2 * a_p * b_p + a_z * b_s
        t = t - (loss_p / loss_s) / (1 + np.sqrt(loss_p ** 2 / loss_s))

        # Clamp to keep a convex combination
        t = np.maximum(np.minimum(t, 1.), 0.)
    return t


def newton_line_search(fin, gin, fout, gout, a, b, rho1, rho2, nits):
    t = 0.5
    for k in range(nits):
        ft = (1 - t) * fin + t * fout
        gt = (1 - t) * gin + t * gout
        transl = rescale_potentials(ft, gt, a, b, rho1, rho2)

        grad = np.sum(a * (fout - fin) * np.exp(-(ft + transl) / rho1)) \
               + np.sum(b * (gout - gin) * np.exp(-(gt - transl) / rho2)) \
            # + 10**(-5-k) / t - 10**(-5-k) / (1-t)
        hess = -np.sum(
            a * (fout - fin) ** 2 * np.exp(-(ft + transl) / rho1)) / rho1 \
               - np.sum(
            b * (gout - gin) ** 2 * np.exp(-(gt - transl) / rho2)) / rho2 \
            # - 10**(-5-k) / t**2 - 10**(-5-k) / (1-t)**2
        t = t + (1 / (1 + np.abs(hess))) * (grad / hess)
        # assert (t>0.) and (t<1.)
    t = np.maximum(np.minimum(t, 1.), 0.)
    return t


#######################
# GREEDY INITIALIZATION
#######################

def init_greed_uot(a, b, x, y, p, rho1, rho2=None):
    if rho2 is None:
        rho2 = rho1

    # Compute balanced potentials
    _, _, _, fb, gb, _ = solve_ot(a / np.sum(a), b / np.sum(b), x, y, p)

    # # Compute lazy potential
    # def cost(i, j):
    #     return np.abs(x[i] - y[j]) ** p
    #
    # spt = compute_shortest_path_support(cost, x.shape[0], y.shape[0])
    # fc, gc = build_potential_from_support(spt, a, x, b, y, rho1, rho2, p)

    # # Output best convex combination
    # t = homogeneous_line_search(fb, gb, fc, gc, a, b, rho1, rho2, nits=3)
    # ft = (1 - t) * fb + t * fc
    # gt = (1 - t) * gb + t * gc
    return fb, gb


def compute_shortest_path_support(func, n, m):
    spt = []
    i, j = 0, 0
    while (i < n - 1) and (j < m - 1):
        c12 = func(i, j + 1)
        c21 = func(i + 1, j)
        c22 = func(i + 1, j + 1)
        if (c22 < c12) and (c22 < c21):
            i = i + 1
            j = j + 1
            spt.append((1, 1))
        elif c12 < c21:
            j = j + 1
            spt.append((0, 1))
        else:
            i = i + 1
            spt.append((1, 0))

    if i == n - 1:
        while j < m - 1:
            j = j + 1
            spt.append((0, 1))

    if j == m - 1:
        while i < n - 1:
            i = i + 1
            spt.append((1, 0))
    return spt


def build_potential_from_support(spt, a, x, b, y, p, rho1, rho2):
    """
    Assuming the optimal support of the transport plan to be known, computes
    the associated optimal dual potentials.

    Parameters
    ----------
    spt: list of pairs of int, of form (1,0), (0,1) or (1,1).
    Describes the moves in coordinates in the transport plan's support.

    a: vector of length n with positive entries

    b: vector of length m with positive entries

    x: vector of real of length n

    y: vector of real of length m

    p: real, should >= 1


    Returns
    -------
    f: dual vector of length n

    g: dual vector of length m
    """
    i0, j0 = 0, 0
    i, j = 0, 0
    f, g = np.zeros_like(a), np.zeros_like(b)
    g[0] = np.abs(x[0] - y[0]) ** p
    for (ip, jp) in spt:
        if (ip == 1) and (jp == 1):
            i = i + ip
            j = j + jp
            # Rescale the finished connected component of support
            transl = rescale_potentials(f[i0:i], g[j0:j], a[i0:i], b[j0:j],
                                        rho1, rho2)
            # Check that translation is admissible
            transl = check_feasibility(i0, j0, transl, f, g, x, y, p)
            f[i0:i], g[j0:j] = f[i0:i] + transl, g[j0:j] - transl
            g[j] = np.abs(x[i] - y[j]) ** p
            i0, j0 = i, j
        elif (jp == 0) and (ip == 1):
            i = i + ip
            j = j + jp
            f[i] = np.abs(x[i] - y[j]) ** p - g[j]
        elif (ip == 0) and (jp == 1):
            i = i + ip
            j = j + jp
            g[j] = np.abs(x[i] - y[j]) ** p - f[i]
        else:
            raise Exception
    transl = rescale_potentials(f[i0:i + 1], g[j0:j + 1], a[i0:i + 1],
                                b[j0:j + 1], rho1, rho2)
    transl = check_feasibility(i0, j0, transl, f, g, x, y, p)
    f[i0:i + 1], g[j0:j + 1] = f[i0:i + 1] + transl, g[j0:j + 1] - transl
    return f, g


def check_feasibility(i, j, transl, f, g, x, y, p):
    if (i == 0) or (j == 0):
        return transl
    else:
        gap01 = np.abs(x[i - 1] - y[j]) ** p - f[i - 1] - g[j]
        gap10 = np.abs(x[i] - y[j - 1]) ** p - f[i] - g[j - 1]
        return np.minimum(gap10, np.maximum(transl, -gap01))
