import numpy as np
import matplotlib.pyplot as plt
import time as time
import os
import torch

from fastuot.uot1d import rescale_potentials, solve_ot, solve_uot
from fastuot.torch_sinkhorn import h_sinkhorn_loop
from fastuot.torch_sinkhorn import rescale_potentials as translate_pot
from utils_examples import generate_synthetic_measure

rc = {"pdf.fonttype": 42, 'text.usetex': True,
      'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(rc)

assert torch.cuda.is_available()

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + 'bench_fw_sink/'
if not os.path.isdir(path):
    os.mkdir(path)



# TODO: make plot with projected WOT data as well
if __name__ == '__main__':
    compute_data = False

    rho = 1.
    p = 2.
    nits = 1000
    nbeg = 0
    N, M = 200, 210
    a, x, b, y = generate_synthetic_measure(N, M)
    C = np.abs(x[:, None] - y[None, :]) ** p

    list_eps = [-2., -2.5, -3.]

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        # Compute unregularized exact dual potential as reference
        _, _, _, fr, _, _ = solve_uot(a, b, x, y, p, rho, niter=20000)
        np.save(path + "ref_pot_maxiter.npy", fr)
        print("Computed reference potential")

        #######################################################################
        # Bench FW-UOT
        #######################################################################
        f, g = np.zeros_like(a), np.zeros_like(b)
        transl = rescale_potentials(f, g, a, b, rho, rho)
        f, g = f + transl, g - transl
        time_fw, acc_fw = [], []
        for k in range(nits):
            t0 = time.time()
            A = np.exp(-f / rho) * a
            B = np.exp(-g / rho) * b

            # update
            I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
            gamma = 2. / (2. + k)  # fixed decaying weights
            f = (1 - gamma) * f + gamma * fs
            g = (1 - gamma) * g + gamma * gs
            transl = rescale_potentials(f, g, a, b, rho, rho)
            f, g = f + transl, g - transl
            time_fw.append(time.time() - t0)
            acc_fw.append(np.amax(np.abs(f - fr)))
        np.save(path + f"time_fw.npy", np.array(time_fw))
        np.save(path + f"err_fw.npy", np.array(acc_fw))

        #######################################################################
        # Bench Sinkhorn GPU
        #######################################################################
        # Convert to cuda tensors
        at = torch.from_numpy(a).cuda()
        bt = torch.from_numpy(b).cuda()
        xt = torch.from_numpy(x).cuda()
        yt = torch.from_numpy(y).cuda()
        Ct = torch.from_numpy(C).cuda()
        frt = torch.from_numpy(fr).cuda()
        # Runover values of entropic regularization
        for k, logeps in enumerate(list_eps):
            eps = 10 ** logeps
            ft, gt = torch.zeros_like(at), torch.zeros_like(bt)
            time_sink, acc_sink = [], []
            for k in range(nits):
                t0 = time.time()
                ft, gt = h_sinkhorn_loop(ft, at, bt, Ct, eps, rho)
                time_sink.append(time.time() - t0)
                transl = translate_pot(ft, gt, at, bt, rho, rho)
                acc_sink.append((ft + transl - frt).abs().max().data.item())
            np.save(path + f"time_sink_eps{logeps}.npy",
                    np.array(time_sink))
            np.save(path + f"err_sink_eps{logeps}.npy",
                    np.array(acc_sink))

    ###########################################################################
    # Plots
    ###########################################################################
    # Compute median time of loop for rescaling x-axis
    list_time = []
    list_time.append(np.median(np.load(path + f"time_fw.npy")))
    for k, logeps in enumerate(list_eps):
        list_time.append(np.median(
            np.load(path + f"time_sink_eps{logeps}.npy")))

    # Aggregating all convergence accuracies and plotting
    list_acc = []
    list_acc.append(np.load(path + f"err_fw.npy"))
    for k, logeps in enumerate(list_eps):
        list_acc.append(
            np.load(path + f"err_sink_eps{logeps}.npy"))

    colors = ['cornflowerblue', 'lightcoral', 'indianred', 'firebrick']
    labels = ['FW'] \
             + [r'TI, $\log\varepsilon$='+str(x) for x in list_eps]

    plt.figure(figsize=(4, 2.5))
    for k, t in enumerate(list_time):
        plt.plot(t * np.arange(1, nits + 1), list_acc[k], c=colors[k],
                  label=labels[k])

    plt.xlabel('Time', fontsize=15)
    plt.ylabel('$\|f_t - f^*\|_\infty$', fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend(fontsize=9, ncol=2, handlelength=1.3, columnspacing=0.5,
               loc=(0.01, 0.01))
    # plt.legend(fontsize=9, ncol=2, handlelength=1.3, columnspacing=0.5,
    #            loc=(0.01, 0.01))
    plt.tight_layout()
    plt.savefig(path + 'plot_bench_fw_sink+0.pdf')
    plt.show()

