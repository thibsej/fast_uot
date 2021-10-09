import numpy as np
import matplotlib.pyplot as plt
import time as time
import os

from fastuot.numpy_sinkhorn import sinkhorn_loop
from fastuot.numpy_sinkhorn import homogeneous_loop as numpy_loop


def gauss(grid, mu, sig):
    return np.exp(-0.5* ((grid-mu) / (sig))**2)


def normalize(x):
    return x / np.sum(x)


def generate_measure(N):
    x = np.linspace(0.2, 0.4, num=N)
    a = np.zeros_like(x)
    a[:N//2] = 2.
    a[N//2:] = 3.
    y = np.linspace(0.45, 0.95, num=N)
    a = normalize(a)
    b = normalize(gauss(y, 0.6, 0.03)
                  + gauss(y, 0.7, 0.03)
                  + gauss(y, 0.8, 0.03))
    return a, x, b, y

if __name__ == '__main__':
    eps_l = [.1, 1.]
    N = 100
    a, x, b, y = generate_measure(N)
    C = (x[:, None] - y[None, :])**2
    scale = [-3., -2.5, -2., -1.5, -1., -0.5, -0., 0.5, 1., 1.5, 2., 2.5, 3.]
    colors = ['r', 'b', 'g', 'c']

    for r in range(len(eps_l)):
        eps = eps_l[r]
        rate_s, rate_ti = [], []
        for s in scale:
            rhot = 10**s
            # Compute reference
            fr, gr = np.zeros_like(a), np.zeros_like(b)
            for i in range(50000):
                f_tmp = fr.copy()
                fr, gr = numpy_loop(fr, a, b, C, eps, rhot)
                if np.amax(np.abs(fr - f_tmp)) < 1e-14:
                    print(np.amax(np.abs(fr - f_tmp)))
                    break

            # Compute error for Sinkhorn
            err_s = []
            f, g = np.zeros_like(a), np.zeros_like(b)
            for i in range(50000):
                f_tmp = f.copy()
                f, g = sinkhorn_loop(f, a, b, C, eps, rhot)
                err_s.append(np.amax(np.abs(f - f_tmp)))
                if np.amax(np.abs(f - f_tmp)) < 1e-12:
                    break
            err_s = np.log10(np.array(err_s))
            err_s = err_s[1:] - err_s[:-1]
            rate_s.append(np.median(err_s))

            # Compute error for TI-Sinkhorn
            err_ti = []
            f, g = np.zeros_like(a), np.zeros_like(b)
            for i in range(50000):
                f_tmp = f.copy()
                f, g = numpy_loop(f, a, b, C, eps, rhot)
                err_ti.append(np.amax(np.abs(f - f_tmp)))
                if np.amax(np.abs(f - f_tmp)) < 1e-12:
                    break
            err_ti = np.log10(np.array(err_ti))
            err_ti = err_ti[1:] - err_ti[:-1]
            rate_ti.append(np.median(err_ti))

        # Plot results
        scale = np.array(scale)
        rate_ti = np.array(rate_ti)
        rate_s = np.array(rate_s)
        rate_th = 2 * np.log10(1 + (eps / rhot))
        rate_th = 0.
        plt.plot(scale, rate_s + rate_th, c=colors[r], linestyle='dashed',
                 label=f'S, $\epsilon$={eps}')
        plt.plot(scale, rate_ti + rate_th, c=colors[r],
                 label=f'TI, $\epsilon$={eps}')
        plt.vlines(np.log10(eps), -6., 0., colors=colors[r], linestyles='dotted')

    plt.legend()
    plt.show()

