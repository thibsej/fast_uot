import matplotlib.pyplot as plt
import numpy as np
import os

# TODO: Update code with new refactored method and folder tree
path = os.getcwd() + "/output/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path + "/paper/"):
    os.mkdir(path + "/paper/")
if not os.path.isdir(path + "/ratekl/"):
    os.mkdir(path + "/ratekl/")

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True,
      'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(rc)

compute_data = False

eps_l = [-2., -1., 0.]
N = 50
rho_scale = np.arange(-3., 3.5, 0.5)
colors = ['', 'cornflowerblue', 'indianred']
markers = ['x', 'o', 'v']
markevery = 2
###########################################################################
    # Make plots
    ###########################################################################
p = 0.97
f, axes = plt.subplots(1, 2, figsize=(p * 8, p * 4), sharey=True)

rho_scale = 10 ** np.load(path + "/ratekl/" + f"rho_scale.npy")

for r in range(1, len(eps_l)):
    epst = 10 ** eps_l[r]
    rate_s = np.load(path + "/ratekl/" + f"rate_f_sinkhorn_kl_eps{epst}.npy")
    rate_ti = np.load(path + "/ratekl/" + f"rate_g_sinkhorn_kl_eps{epst}.npy")
    rate_f = np.load(path + "/ratekl/" + f"rate_h_sinkhorn_kl_eps{epst}.npy")
    axes[0].plot(rho_scale, 10 ** rate_s, c=colors[r], linestyle='dashed',
             label=f'$S,\,\epsilon=${epst}', marker=markers[0], markevery=markevery)
    axes[0].plot(rho_scale, 10 ** rate_ti, c=colors[r], linestyle='dotted',
             label=f'$TI,\,\epsilon=${epst}', marker=markers[1], markevery=markevery)
    axes[0].plot(rho_scale, 10 ** rate_f, c=colors[r],
             label=f'$F,\,\epsilon=${epst}', marker=markers[2], markevery=markevery)


rho_scale = 10 ** np.load(path + "/rateberg/" + f"rho_scale.npy")
for r in range(1, len(eps_l)):
    epst = 10**eps_l[r]
    rate_f = np.load(path + "/rateberg/" + f"rate_f_sinkhorn_berg_eps{epst}.npy")
    rate_g = np.load(path + "/rateberg/" + f"rate_g_sinkhorn_berg_eps{epst}.npy")
    axes[1].plot(rho_scale, 10 ** rate_f, c=colors[r], linestyle='dashed',
                 marker=markers[0], markevery=markevery,
                 label=r'$\mathcal{F}_{\varepsilon}$, $\varepsilon=%.1f$' % epst)
    axes[1].plot(rho_scale, 10 ** rate_g, c=colors[r], linestyle='dotted',
                 marker=markers[1], markevery=markevery,
             label=r'$\mathcal{G}_{\varepsilon}$, $\varepsilon=%.1f$' % epst)
    axes[1].plot([], [],  c=colors[r],
                 marker=markers[2], markevery=markevery,
             label=r'$\mathcal{H}_{\varepsilon}$, $\varepsilon=%.1f$' % epst)

axes[1].legend(fontsize=11, ncol=2, columnspacing=0.5, handlelength=1.3,
               loc=(.25, .02))
for ax in axes:
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-6, 1.5])
    ax.set_xlabel('Marginal parameter $\\rho$', fontsize=15)
axes[0].set_title('KL entropy', fontsize=18)
axes[1].set_title('Berg entropy', fontsize=18)
axes[0].set_ylabel('Contraction rate', fontsize=15)

plt.tight_layout()
plt.savefig(path + "/paper/" + 'plot_log_contraction_rate_kl_berg_v2.pdf')