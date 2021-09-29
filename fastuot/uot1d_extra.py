import numpy as np
from uot1d import rescale_potentials

def sparsify_support(spt, down=True):
    spts = []
    k = 0
    while k < (len(spt) - 1):
        a, b = spt[k], spt[k + 1]
        if down and (a == (0,1)) and (b == (1,0)):
            spts.append((1,1))
            k = k+2
        if (not down) and (a == (1,0)) and (b == (0,1)):
            spts.append((1, 1))
            k = k + 2
        else:
            spts.append(a)
            k = k + 1
    if k == len(spt) - 1:
        spts.append(b)
    return spts


def compute_shortest_path_support(func, n, m, diagonal=False):
    spt = []
    i, j = 0, 0
    while (i < n - 1) and (j < m - 1):
        c12 = func(i, j + 1)
        c21 = func(i + 1, j)
        c22 = func(i + 1, j + 1)
        if diagonal and (c22 < c12) and (c22 < c21):
            i = i + 1
            j = j + 1
            spt.append((1, 1))
        elif c12 > c21:
        # elif c12 < c21:
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