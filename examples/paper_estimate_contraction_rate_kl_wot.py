import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import wot

from fastuot.numpy_sinkhorn import sinkhorn_loop, faster_loop, balanced_loop
from fastuot.numpy_sinkhorn import homogeneous_loop as numpy_loop
from fastuot.uot1d import hilbert_norm, rescale_potentials

path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "/paper/"):
    os.mkdir(path + "/paper/")
if not os.path.isdir(path + "/rateklwot/"):
    os.mkdir(path + "/rateklwot/")

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
plt.rcParams.update(rc)
# TODO: refactor with other xp on KL

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

def load_wot_data():
    gene_set_scores = pd.read_csv('data/gene_set_scores.csv', index_col=0)
    proliferation = gene_set_scores['Cell.cycle']
    apoptosis = gene_set_scores['Apoptosis']

    # apply logistic function to transform to birth rate and death rate
    def logistic(x, L, k, x0=0):
        f = L / (1 + np.exp(-k * (x - x0)))
        return f

    def gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width):
        return beta_min + logistic(p, L=beta_max - beta_min, k=4 / width,
                                   x0=center)

    def beta(p, beta_max=1.7, beta_min=0.3, pmax=1.0, pmin=-0.5, center=0.25):
        return gen_logistic(p, beta_max, beta_min, pmax, pmin, center,
                            width=0.5)

    def delta(a, delta_max=1.7, delta_min=0.3, amax=0.5, amin=-0.4,
              center=0.1):
        return gen_logistic(a, delta_max, delta_min, amax, amin, center,
                            width=0.2)

    birth = beta(proliferation)
    death = delta(apoptosis)

    # growth rate is given by
    gr = np.exp(birth - death)
    growth_rates_df = pd.DataFrame(index=gene_set_scores.index,
                                   data={'cell_growth_rate': gr})
    growth_rates_df.to_csv('data/growth_gs_init.txt')

    VAR_GENE_DS_PATH = 'data/ExprMatrix.var.genes.h5ad'
    CELL_DAYS_PATH = 'data/cell_days.txt'
    SERUM_CELL_IDS_PATH = 'data/serum_cell_ids.txt'
    CELL_GROWTH_PATH = 'data/growth_gs_init.txt'

    # load data
    adata = wot.io.read_dataset(VAR_GENE_DS_PATH,
                                obs=[CELL_DAYS_PATH, CELL_GROWTH_PATH],
                                obs_filter=SERUM_CELL_IDS_PATH)
    # print(adata.shape)
    # print(type(adata))
    # print(adata)
    # print(adata.X[0])
    ot_model = wot.ot.OTModel(adata, epsilon=0.05, lambda1=1, lambda2=50)
    t0, t1 = 7, 8
    ds = ot_model.matrix

    p0_indices = ds.obs[ot_model.day_field] == float(t0)
    p1_indices = ds.obs[ot_model.day_field] == float(t1)

    p0 = ds[p0_indices, :]
    p1 = ds[p1_indices, :]
    local_pca = ot_model.ot_config.pop('local_pca', None)
    eigenvals = None
    if local_pca is not None and local_pca > 0:
        # pca, mean = wot.ot.get_pca(local_pca, p0.X, p1.X)
        # p0_x = wot.ot.pca_transform(pca, mean, p0.X)
        # p1_x = wot.ot.pca_transform(pca, mean, p1.X)
        p0_x, p1_x, pca, mean = wot.ot.compute_pca(p0.X, p1.X, local_pca)
        eigenvals = np.diag(pca.singular_values_)
    else:
        p0_x = p0.X
        p1_x = p1.X
    C = ot_model.compute_default_cost_matrix(p0_x, p1_x, eigenvals)
    a, b = np.ones(C.shape[0]) / C.shape[0], np.ones(C.shape[1]) / C.shape[1]
    return a, b, C


if __name__ == '__main__':
    compute_data = False
    wot_data = True
    # raise SystemExit

    if wot_data:
        a, b, C = load_wot_data()
    else:
        N = 50
        a, x, b, y = generate_measure(N)
        C = (x[:, None] - y[None, :])**2
    
    eps_l = [-1., 0.]
    # eps_l = [0.]
    rho_scale = np.arange(-3., 3.5, 0.5)
    colors = [(1., 0., 0., 1.), (.5, 0., .5, 1.), (0., 0., 1., 1.)]

    ###########################################################################
    # Generate data plots
    ###########################################################################
    if compute_data:
        np.save(path + "/rateklwot/" + f"rho_scale.npy", rho_scale)
        for r in range(len(eps_l)):
            epst = 10 ** eps_l[r]
            rate_s, rate_ti, rate_f = [], [], []
            for s in rho_scale:
                rhot = 10**s
                print(f"(eps, rho) = {(epst, rhot)}")
                # Compute reference
                fr, gr = np.zeros_like(a), np.zeros_like(b)
                for i in range(5000):
                    f_tmp = fr.copy()
                    if epst <= rhot:
                        fr, gr = numpy_loop(fr, a, b, C, epst, rhot)
                    else:
                        fr, gr = sinkhorn_loop(fr, a, b, C, epst, rhot)
                    if np.amax(np.abs(fr - f_tmp)) < 1e-14:
                        break

                # Compute error for F - Sinkhorn
                err_s = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(300):
                    f_tmp = f.copy()
                    f, g = sinkhorn_loop(f, a, b, C, epst, rhot)
                    t = rescale_potentials(f, g, a, b, rhot)
                    err_s.append(np.amax(np.abs(f + t - fr)))
                    if np.amax(np.abs(f + t - fr)) < 1e-12:
                        break
                err_s = np.log10(np.array(err_s))
                err_s = err_s[1:] - err_s[:-1]
                rate_s.append(np.median(err_s))

                # Compute error for G - Sinkhorn
                err_ti = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(300):
                    f_tmp = f.copy()
                    f, g = numpy_loop(f, a, b, C, epst, rhot)
                    t = rescale_potentials(f, g, a, b, rhot)
                    err_ti.append(np.amax(np.abs(f + t - fr)))
                    if np.amax(np.abs(f + t - fr)) < 1e-12:
                        break
                err_ti = np.log10(np.array(err_ti))
                err_ti = err_ti[1:] - err_ti[:-1]
                rate_ti.append(np.median(err_ti))

                # Compute error for H - Sinkhorn
                err_f = []
                f, g = np.zeros_like(a), np.zeros_like(b)
                for i in range(300):
                    f_tmp = f.copy()
                    f, g = faster_loop(f, a, b, C, epst, rhot)
                    t = rescale_potentials(f, g, a, b, rhot)
                    err_f.append(np.amax(np.abs(f + t - fr)))
                    if np.amax(np.abs(f + t - fr)) < 1e-12:
                        break
                err_f = np.log10(np.array(err_f))
                err_f = err_f[1:] - err_f[:-1]
                rate_f.append(np.median(err_f))


            # Plot results
            np.save(
                path + "/rateklwot/" + f"rate_f_sinkhorn_kl_eps{epst}.npy",
                np.array(rate_s))
            np.save(
                path + "/rateklwot/" + f"rate_g_sinkhorn_kl_eps{epst}.npy",
                np.array(rate_ti))
            np.save(
                path + "/rateklwot/" + f"rate_h_sinkhorn_kl_eps{epst}.npy",
                np.array(rate_f))


    ###########################################################################
    # Make plots
    ###########################################################################
    p = 0.97
    colors = ['cornflowerblue', 'indianred']
    markers = ['x', 'o', 'v']
    markevery = 1
    f, ax = plt.subplots(1, 1, figsize=(p * 5, p * 4))

    rho_scale = 10 ** np.load(path + "/rateklwot/" + f"rho_scale.npy")

    for r in range(0, len(eps_l)):
        epst = 10 ** eps_l[r]
        rate_s = np.load(
            path + "/rateklwot/" + f"rate_f_sinkhorn_kl_eps{epst}.npy")
        rate_ti = np.load(
            path + "/rateklwot/" + f"rate_g_sinkhorn_kl_eps{epst}.npy")
        rate_f = np.load(
            path + "/rateklwot/" + f"rate_h_sinkhorn_kl_eps{epst}.npy")
        ax.plot(rho_scale, 10 ** rate_s, c=colors[r], linestyle='dashed',
                     marker=markers[0], markevery=markevery,
                     label=r'$\mathcal{F}_{\varepsilon}$, $\varepsilon=%.1f$' % epst)
        ax.plot(rho_scale, 10 ** rate_ti, c=colors[r], linestyle='dotted',
                     marker=markers[1], markevery=markevery,
                     label=r'$\mathcal{G}_{\varepsilon}$, $\varepsilon=%.1f$' % epst)
        ax.plot(rho_scale, 10 ** rate_f, c=colors[r],
                     marker=markers[2], markevery=markevery,
                     label=r'$\mathcal{H}_{\varepsilon}$, $\varepsilon=%.1f$' % epst)

    ax.legend(fontsize=11, ncol=3, columnspacing=0.5, handlelength=1.3,
                   loc=(.02, .02))

    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Marginal parameter $\\rho$', fontsize=15)
    ax.set_title('KL entropy', fontsize=18)
    ax.set_ylabel('Contraction rate', fontsize=15)

    plt.tight_layout()
    plt.savefig(path + "/paper/" + 'plot_log_contraction_rate_kl_fast_wot.pdf')
    plt.show()