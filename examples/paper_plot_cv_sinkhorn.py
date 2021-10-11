import numpy as np
import matplotlib.pyplot as plt
import os

from fastuot.numpy_sinkhorn import sinkhorn_loop
from fastuot.numpy_sinkhorn import homogeneous_loop as numpy_loop, faster_loop

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/paper/"
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)


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
    N = 200
    a, x, b, y = generate_measure(N)
    C = (x[:, None] - y[None, :]) ** 2
    eps, rho = 0.001, .1
    Nits_inf, Nits = 5000, 500

    scale = [10., 1., 0.1]
    col = ['b', 'r', 'g', 'm']
    lw = 2.
    plt.figure(figsize=(5, 4))
    for p in range(len(scale)):
        epst, rhot = scale[p] * eps, scale[p] * rho

        # Compute reference
        fr, gr = np.zeros_like(a), np.zeros_like(b)
        for i in range(Nits_inf):
            fr, gr = numpy_loop(fr, a, b, C, epst, rhot)

        # compute norm for sinkhorn
        f, g = np.zeros_like(a), np.zeros_like(b)
        err_sink = []
        for i in range(Nits):
            f, g = sinkhorn_loop(f, a, b, C, epst, rhot)
            err_sink.append(np.amax(np.abs(f - fr)))

        # compute norm for sinkhorn
        f, g = np.zeros_like(a), np.zeros_like(b)
        err_hom = []
        for i in range(Nits):
            f, g = numpy_loop(f, a, b, C, epst, rhot)
            err_hom.append(np.amax(np.abs(f - fr)))

        plt.plot(np.log10(np.array(err_sink)), color=col[p], linestyle='dashed',
                 label=f'$S, t=${scale[p]}', linewidth=lw)
        plt.plot(np.log10(np.array(err_hom)), color=col[p],
                 label=f'$TI, t=${scale[p]}', linewidth=lw)

    plt.xlabel('$Iterations$', fontsize=16)
    plt.ylabel('$\log_{10}\|f_t - f^*\|_\infty$', fontsize=16)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path + 'plot_sinkhorn_ratio_fixed.pdf')
    plt.show()
