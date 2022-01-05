import os

import numpy as np
import matplotlib.pyplot as plt

from fastuot.uot1d import solve_ot, rescale_potentials
from utils_examples import normalize, gauss

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/paper/"
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True,
      'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(rc)


def generate_sample_measure2():
    nsampl = 1000
    grid = np.linspace(0, 1, nsampl)
    a = np.zeros_like(grid)
    a[200:300] = 2.
    a[300:400] = 3.
    a = normalize(a)
    b = normalize(gauss(grid, 0.6, 0.03)
                  + gauss(grid, 0.7, 0.03)
                  + gauss(grid, 0.8, 0.03))
    return a, b, grid


def plot_figstep(k):
    plt.figure(figsize=(0.5 * 12, 0.5 * 5))
    plt.plot(grid, a, c='r', linestyle='dashed', label='input 1')
    plt.plot(grid, b, c='b', linestyle='dashed', label='input 2')
    plt.plot(grid, Ar, c='m', linestyle='dashed', label='target 1')
    plt.plot(grid, Br, c='c', linestyle='dashed', label='target 2')
    plt.plot(grid, A, c='r', label='marg 1', )
    plt.plot(grid, B, c='b', label='marg 2', )
    plt.fill_between(grid, Ar, A, color=(0.95,0.55,0.55,0.3))
    plt.fill_between(grid, Br, B, color=(0.55,0.55,0.95,0.3))
    plt.title(f'Iterate {k}', fontsize=16)
    plt.ylim(0.0, 0.0065)
    plt.xlim(0.15, 0.91)
    plt.axis('off')
    if k==0:
        plt.legend(loc=(0.4, 0.72), fontsize=12, ncol=3, columnspacing=0.5,
                   handlelength=1.3)
    plt.tight_layout()
    plt.savefig(path + f'sequence_marginals_fw_iter_{k}.pdf')
    plt.show()

def fw_step(f, g, a, b, rho1, rho2, k):
    transl = rescale_potentials(f, g, a, b, rho, rho)
    f, g = f + transl, g - transl
    A = np.exp(-f / rho1) * a
    B = np.exp(-g / rho2) * b

    # update
    I, J, P, fs, gs, _ = solve_ot(A, B, grid, grid, p)
    gamma = 2. / (2. + k)  # fixed decaying weights
    f = f + gamma * (fs - f)
    g = g + gamma * (gs - g)
    return f, g, A, B


if __name__ == '__main__':
    single_pic = False
    a, b, grid = generate_sample_measure2()

    # params
    p = 2.
    rho = 0.1
    niter = 4

    fr, gr = np.zeros_like(a), np.zeros_like(b)
    for k in range(50000):
        fr, gr, Ar, Br = fw_step(fr, gr, a, b, rho, rho, k)

    if single_pic:
        colors_a = ['darkred', 'firebrick', 'indianred', 'lightcoral']
        colors_b = ['mediumblue', 'royalblue', 'steelblue', 'cornflowerblue']
        plt.figure(figsize=(0.5 * 12, 0.5 * 5))

        f, g = np.zeros_like(a), np.zeros_like(b)
        plt.plot(grid, a, c=colors_a[0], label='input $\\alpha$')
        plt.plot(grid, b, c=colors_b[0], label='input $\\beta$')
        f, g, A, B = fw_step(f, g, a, b, rho, rho, 0)
        for k in range(niter):
            print(k + 1)
            f, g, A, B = fw_step(f, g, a, b, rho, rho, k + 1)
            plt.plot(grid, A, c=colors_a[k + 1],
                     linestyle='dashed')
            plt.plot(grid, B, c=colors_b[k + 1],
                     linestyle='dashed')

        plt.plot(grid, Ar, c=colors_a[-1],
                 label='optimal $\\bar{\\alpha}$')
        plt.plot(grid, Br, c=colors_b[-1],
                 label='optimal $\\bar{\\beta}$')

        plt.ylim(0.0, 0.0065)
        plt.xlim(0.15, 0.91)
        plt.axis('off')
        plt.legend(fontsize=12, ncol=2, columnspacing=0.5,
                   handlelength=1.3)

        plt.tight_layout()
        plt.savefig(path + f'sequence_marginals_fw_onepic.pdf')
        plt.show()
    else:
        f, g = np.zeros_like(a), np.zeros_like(b)
        for k in range(niter):
            f, g, A, B = fw_step(f, g, a, b, rho, rho, k)
            plot_figstep(k)
