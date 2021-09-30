import numpy as np
import matplotlib.pyplot as plt
import time as time
import os
import torch

from fastuot.uot1d import rescale_potentials, solve_ot
from fastuot.numpy_sinkhorn import sinkhorn_loop, fast_homogeneous_loop
from fastuot.numpy_sinkhorn import homogeneous_loop as numpy_loop
from fastuot.torch_sinkhorn import homogeneous_loop as torch_loop
from fastuot.cvxpy_uot import dual_via_cvxpy

assert torch.cuda.is_available()

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/plots_comparison/"
if not os.path.isdir(path):
    os.mkdir(path)


if __name__ == '__main__':
    Ntrials = 1
    rho = .1
    p = 2.
    nits = 5000
    nbeg = 0
    N, M = 200, 210

    acc_fw = np.zeros((Ntrials, nits))
    time_fw = np.zeros((Ntrials, nits))

    list_eps_lse = [-2., -2.5, -3., -3.5]
    acc_lse = np.zeros((len(list_eps_lse), Ntrials, nits))
    time_lse = np.zeros((len(list_eps_lse), Ntrials, nits))

    list_eps_gpu = [-2., -2.5, -3., -3.5]
    acc_gpu = np.zeros((len(list_eps_lse), Ntrials, nits))
    time_gpu = np.zeros((len(list_eps_lse), Ntrials, nits))

    list_eps_exp = [-0.]
    acc_exp = np.zeros((len(list_eps_exp), Ntrials, nits))
    time_exp = np.zeros((len(list_eps_exp), Ntrials, nits))

    cmap = plt.get_cmap('autumn')

    for i in range(Ntrials):
        print(f"Trial {i}")
        np.random.seed(i)
        a, b = np.random.exponential(size=N), np.random.exponential(size=M)
        a, b = a / np.sum(a), b / np.sum(b)
        x, y = np.sort(np.random.normal(size=N)), \
               np.sort(np.random.normal(loc=0.2, size=M))
        C = np.abs(x[:, None] - y[None, :]) ** p

        _, _, f, _ = dual_via_cvxpy(a, b, x, y, p, rho, cpsolv='SCS',
                                    tol=1e-7, niter=500000)
        fr = f.value
        print("Computed reference potential")

        #######################################################################
        # Bench FW-UOT
        #######################################################################
        f, g = np.zeros_like(a), np.zeros_like(b)
        dual_val = []
        for k in range(nits):
            t0 = time.time()
            transl = rescale_potentials(f, g, a, b, rho, rho)
            f, g = f + transl, g - transl
            A = np.exp(-f / rho) * a
            B = np.exp(-g / rho) * b

            # update
            I, J, P, fs, gs, _ = solve_ot(A, B, x, y, p)
            gamma = 2. / (2. + k)  # fixed decaying weights
            f = (1 - gamma) * f + gamma * fs
            g = (1 - gamma) * g + gamma * gs
            time_fw[i, k] = time.time() - t0
            acc_fw[i, k] = np.log10(np.sum(a * np.abs(f - fr)))
            # acc_fw[i, k] = np.log10(np.amax(np.abs(f - fr)))
        plt.plot(np.mean(time_fw[:, :]) * np.arange(nits),
                 np.mean(acc_fw, axis=0), c='b', label='fw')

        #######################################################################
        # Bench Sinkhorn CPU
        #######################################################################
        for j in range(len(list_eps_lse)):
            eps = 10 ** list_eps_lse[j]
            f, g = np.zeros_like(a), np.zeros_like(b)
            dual_val = []
            for k in range(nits):
                t0 = time.time()
                f, g = numpy_loop(f, a, b, C, eps, rho, rho2=None)
                time_lse[j, i, k] = time.time() - t0
                # acc_lse[j, i, k] = np.log10(np.amax(np.abs(f - fr)))
                acc_lse[j, i, k] = np.log10(np.sum(a * np.abs(f - fr)))

            fpt, _ = sinkhorn_loop(f, a, b, C, eps, rho, rho2=None)
            print(np.amax(np.abs(f - fpt)))

        for j in range(len(list_eps_lse)):
            k = list_eps_lse[j]
            eps = 10 ** k
            plt.plot(np.mean(time_lse[j, :, :]) * np.arange(nits),
                     np.mean(acc_lse[j, :, :], axis=0),
                     label=f'Log-CPU ({k})',
                     c=cmap(j / (len(list_eps_lse) + len(list_eps_exp))))

        for j in range(len(list_eps_exp)):
            k = list_eps_exp[j]
            eps = 10 ** k
            K = np.exp(-C / eps)
            u, v = np.ones_like(a), np.ones_like(b)
            dual_val = []
            for k in range(nits):
                t0 = time.time()
                u, v = fast_homogeneous_loop(u, a, b, K, eps, rho)
                time_exp[j, i, k] = time.time() - t0
                f = eps * np.log(u)
                # acc_exp[j, i, k] = np.log10(np.amax(np.abs(f - fr)))
                acc_exp[j, i, k] = np.log10(np.sum(a * np.abs(f - fr)))

        for j in range(len(list_eps_exp)):
            k = list_eps_exp[j]
            eps = 10 ** k
            plt.plot(np.mean(time_exp[j, :, :]) * np.arange(nits),
                     np.mean(acc_exp[j, :, :], axis=0),
                     label=f'Exp-CPU ({k})',
                     c=cmap((j + len(list_eps_lse)) / (len(list_eps_lse) + len(list_eps_exp))))

        #######################################################################
        # Bench Sinkhorn GPU
        #######################################################################
        at = torch.from_numpy(a).cuda()
        bt = torch.from_numpy(b).cuda()
        Ct = torch.from_numpy(C).cuda()
        frt = torch.from_numpy(fr).cuda()
        for j in range(len(list_eps_gpu)):
            eps = 10 ** list_eps_gpu[j]
            ft, gt = torch.zeros_like(at), torch.zeros_like(bt)
            dual_val = []
            for k in range(nits):
                t0 = time.time()
                ft, gt = torch_loop(ft, at, bt, Ct, eps, rho, rho2=None)
                time_gpu[j, i, k] = time.time() - t0
                # acc_lse[j, i, k] = np.log10(np.amax(np.abs(f - fr)))
                acc_gpu[j, i, k] = ((at * (ft - frt).abs()).sum()).log10().data.item()


        for j in range(len(list_eps_lse)):
            k = list_eps_lse[j]
            eps = 10 ** k
            plt.plot(np.mean(time_gpu[j, :, :]) * np.arange(nits),
                     np.mean(acc_gpu[j, :, :], axis=0),
                     label=f'Log-GPU ({k})', linestyle='dashed',
                     c=cmap(j / (len(list_eps_lse) + len(list_eps_exp))))

    plt.legend()
    plt.xlabel('time', fontsize=16)
    plt.ylabel('$\log||f_t - f_*||_\infty$', fontsize=16)
    plt.tight_layout()
    plt.savefig(path + f"bench_comparison_fast_sink_fw_rho{rho}.png")
    plt.show()
