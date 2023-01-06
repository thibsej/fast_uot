import os
import numpy as np
import matplotlib.pyplot as plt

from utils_examples import generate_synthetic_measure

from fastuot.numpy_sinkhorn import f_sinkhorn_loop, h_sinkhorn_loop

path = os.getcwd() + "/output/"
print(path)
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "sinkcv/"
if not os.path.isdir(path):
    os.mkdir(path)

rc = {"pdf.fonttype": 42, 'text.usetex': True,
      'text.latex.preamble': [r'\usepackage{amsmath}',
                              r'\usepackage{amssymb}']}
plt.rcParams.update(rc)

if __name__ == '__main__':
    compute_data = True

    N = 100
    a, x, b, y = generate_synthetic_measure(N, N)
    C = (x[:, None] - y[None, :]) ** 2
    eps_r, rho_r = 0.001, 1.
    func_method = [f_sinkhorn_loop, h_sinkhorn_loop]
    dataname = 'synthetic'
    string_method = ['f', 'h']
    penalty = 'kl'
    t = 1e-2
    scale_l = [1e-0, 1e-2]


    if compute_data:
        for t in scale_l:
            eps, rho = t * eps_r, t * rho_r

            # Compute fixed point
            fr, gr = np.zeros_like(a), np.zeros_like(b)
            for i in range(20000):
                f_tmp = fr.copy()
                fr, gr = h_sinkhorn_loop(fr, a, b, C, eps, rho)
                fr, gr = f_sinkhorn_loop(fr, a, b, C, eps, rho)
                if np.amax(np.abs(fr - f_tmp)) < 1e-15:
                    break
            print("     Estimated fixed point.")

            # Compute error
            for k, (s, loop) in enumerate(zip(string_method, func_method)):
                err = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(2000):
                    f_tmp = f.copy()
                    f, g = loop(f, a, b, C, eps, rho)

                    err.append(np.amax(np.abs(f - fr)))
                    if np.amax(np.abs(f - fr)) < 1e-12:
                        break
                np.save(
                    path + "error_" + s + f"_sinkhorn_{penalty}_eps{eps}_{dataname}.npy",
                    np.array(err))

    ###########################################################################
    # Make plots
    ###########################################################################
    p = 0.9
    colors = ['cornflowerblue', 'indianred']
    markers = ['v', 'o']
    linestyles = ['dashed', 'dotted']
    # labels = ['$\mathcal{F},\,t=$', '$\mathcal{H},\,t=$']
    labels = ['S, t=', 'TI, t=']
    markevery = 150
    f, ax = plt.subplots(1, 1, figsize=(p * 6, p * 4))

    for t, linestyle, marker in zip(scale_l, linestyles, markers):
        eps, rho = t * eps_r, t * rho_r
        for s, label, color in zip(string_method, labels, colors):
            err = np.load(
                path + f"error_" + s + f"_sinkhorn_{penalty}_eps{eps}_{dataname}.npy")
            ax.plot(err, c=color, linestyle=linestyle,
                    label=label + f'{t}',
                    marker=marker, markevery=markevery)

        ax.legend(fontsize=11, ncol=2, columnspacing=0.5, handlelength=2.)

    ax.grid()
    ax.set_yscale('log')
    # ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Number of Iterations', fontsize=18)
    if penalty == 'kl':
        ax.set_title('KL entropy', fontsize=22)
    if penalty == 'berg':
        ax.set_title('Berg entropy', fontsize=22)
    ax.set_ylabel('$|| f_t - f^\star||_{\infty}$', fontsize=18)

    plt.tight_layout()
    plt.savefig(
        path + f'plot_cv_error_{penalty}_{dataname}.pdf')
    plt.show()