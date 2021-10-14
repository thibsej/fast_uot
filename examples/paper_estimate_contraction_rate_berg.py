import numpy as np
import matplotlib.pyplot as plt
import os

from fastuot.numpy_berg import sinkhorn_loop, homogeneous_loop

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/paper/"
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)


def gauss(grid, mu, sig):
    return np.exp(-0.5 * ((grid-mu) / sig) ** 2)


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
    eps_l = [.01, .1, 1.]
    eps_l = [0.1, 1., 10.]
    N = 100
    a, x, b, y = generate_measure(N)
    C = (x[:, None] - y[None, :])**2
    scale = np.arange(-3., 3.5, 0.5)
    # colors = ['r', 'b', 'g', 'c']
    # colors = [(1., 0., 0., 1.), (.67, 0., .33, 1.), (.33, 0., .67, 1.),
    #           (0., 0., 1., 1.)]
    colors = [(1., 0., 0., 1.), (.5, 0., .5, 1.), (0., 0., 1., 1.)]

    plt.figure(figsize=(8, 5))
    for r in range(len(eps_l)):
        epst = eps_l[r]
        rate_s, rate_ti = [], []
        for s in scale:
            rhot = 10**s
            print(f"(eps, rho) = {(epst, rhot)}")
            # Compute reference
            fr, gr = np.zeros_like(a), np.zeros_like(b)
            for i in range(50000):
                f_tmp = fr.copy()
                if epst <= rhot:
                    fr, gr = homogeneous_loop(fr, a, b, C, epst, rhot)
                else:
                    fr, gr = sinkhorn_loop(fr, a, b, C, epst, rhot)
                # print(np.amax(np.abs(fr - f_tmp)))
                if (np.amax(np.abs(fr - f_tmp)) < 1e-15):
                    break

            # Compute error for Sinkhorn
            err_s = []
            f, g = np.zeros_like(a), np.zeros_like(b)
            for i in range(5000):
                f_tmp = f.copy()
                f, g = sinkhorn_loop(f, a, b, C, epst, rhot)
                err_s.append(np.amax(np.abs(f - fr)))
                if np.amax(np.abs(f - fr)) < 1e-12:
                    break
            err_s = np.log10(np.array(err_s))
            err_s = err_s[1:] - err_s[:-1]
            rate_s.append(np.median(err_s))

            # Compute error for TI-Sinkhorn
            err_ti = []
            f, g = np.zeros_like(a), np.zeros_like(b)
            for i in range(5000):
                f_tmp = f.copy()
                f, g = homogeneous_loop(f, a, b, C, epst, rhot)
                err_ti.append(np.amax(np.abs(f - fr)))
                if np.amax(np.abs(f - fr)) < 1e-12:
                    break
            err_ti = np.log10(np.array(err_ti))
            err_ti = err_ti[1:] - err_ti[:-1]
            rate_ti.append(np.median(err_ti))

        # Plot results
        scale = np.array(scale)
        rate_ti = np.array(rate_ti)
        rate_s = np.array(rate_s)
        plt.plot(scale, rate_s, c=colors[r], linestyle='dashed',
                 label=f'$S,\,\epsilon=${epst}')
        plt.plot(scale, rate_ti, c=colors[r],
                 label=f'$TI,\,\epsilon=${epst}')

    plt.xlabel('$\log_{10}(\\rho)$', fontsize=20)
    plt.ylabel('$Log$-$contraction$ $rate$', fontsize=20)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path + 'plot_log_contraction_berg.pdf')
    plt.show()

