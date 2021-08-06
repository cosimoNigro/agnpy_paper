# script to study the effect of the integration grid on the final SED resolution
import sys
import numpy as np
import astropy.units as u
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.targets import SphericalShellBLR
from agnpy.absorption import Absorption
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
from pathlib import Path

# BLR definition
L_disk = 2 * 1e46 * u.Unit("erg s-1")
xi_line = 0.024
R_line = 1.1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

# different resolution
n_mu_list = [50, 100, 200, 300, 400]
n_phi_list = [50, 100, 200, 300, 400]
colors = ["crimson", "dodgerblue", "goldenrod", "lightseagreen", "k"]
# same frequency grid
nu = np.logspace(15, 29, 100) * u.Hz

# show the effect of changing the Lorentz factor and frequency grid
taus_variable_mu = []
labels_variable_mu = []
for n_mu in n_mu_list:
    abs = Absorption(blr, r=1e18 * u.cm, z=1)
    abs.set_mu(n_mu)
    tau = abs.tau(nu)
    label = r"$n_{\mu}=$" + f"{n_mu}, " + r"$n_{\phi}=50,\,n_{\r}=50,\,n_{\nu}=100$"
    taus_variable_mu.append(tau)
    labels_variable_mu.append(label)

# show the effect of changing the azimuth angle grid
taus_variable_phi = []
labels_variable_phi = []
for n_phi in n_phi_list:
    abs = Absorption(blr, r=1e18 * u.cm, z=1)
    abs.set_phi(n_phi)
    tau = abs.tau(nu)
    label = r"$n_{\mu}=100,\,n_{\phi}=" + f"{n_phi}" + r"$,\,n_{\r}=50,\,n_{\nu}=100$"
    taus_variable_mu.append(tau)
    labels_variable_mu.append(label)

# figure
load_mpl_rc()
plt.rcParams["text.usetex"] = True
# gridspec plot setting
fig = plt.figure(figsize=(12, 6), tight_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[2, 1], figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 0], sharex=ax1)
ax4 = fig.add_subplot(spec[1, 1], sharex=ax2, sharey=ax3)
# changing Lorentz factor and frequency grid
for sed, label, color in zip(
    taus_variable_mu[:-1], labels_variable_mu[:-1], colors[:-1]
):
    ax1.loglog(nu, sed, ls="-", color=color, label=label)
# plot the last one as reference
ax1.loglog(
    nu,
    taus_variable_mu[-1],
    ls="--",
    color=colors[-1],
    label=labels_variable_mu[-1],
)
ax1.set_ylabel(r"$\tau_{\gamma\gamma}$")
ax1.legend(loc="best", fontsize=10)
ax1.set_title("abs. on BLR, " + r"$r=10^{18}\,{\rm cm}$")
# changing azimuth angle grid
for sed, label, color in zip(
    taus_variable_phi[:-1], labels_variable_phi[:-1], colors[:-1]
):
    ax2.loglog(nu, sed, ls="-", color=color, label=label)
# plot the last one as reference
ax2.loglog(
    nu, taus_variable_phi[-1], ls="--", color=colors[-1], label=labels_variable_phi[-1]
)
ax2.legend(loc="best", fontsize=10)
ax2.set_title("EC on ring DT, " + r"$r=10^{21}\,{\rm cm}$")
# plot the deviation from the denser SED in the bottom panel
for sed, label, color in zip(
    taus_variable_phi[:-1], taus_variable_phi[:-1], colors[:-1]
):
    deviation = (sed / taus_variable_mu[-1]) - 1
    ax3.semilogx(nu, deviation, ls="-", color=color, label=label)
ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.3, ls=":", color="darkgray")
ax3.axhline(-0.3, ls=":", color="darkgray")
ax3.set_ylim([-0.5, 0.5])
ax3.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax3.set_xlabel(sed_x_label)
ax3.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")
# plot the deviation from the denser SED in the bottom panel
for sed, label, color in zip(
    taus_variable_phi[:-1], labels_variable_phi[:-1], colors[:-1]
):
    deviation = (sed / taus_variable_phi[-1]) - 1
    ax4.semilogx(nu, deviation, ls="-", color=color, label=label)
ax4.grid(False)
ax4.axhline(0, ls="-", color="darkgray")
ax4.axhline(0.2, ls="--", color="darkgray")
ax4.axhline(-0.2, ls="--", color="darkgray")
ax4.axhline(0.3, ls=":", color="darkgray")
ax4.axhline(-0.3, ls=":", color="darkgray")
ax4.set_ylim([-0.5, 0.5])
ax4.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax4.set_xlabel(sed_x_label)
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_appendix_B_absorption.png")
fig.savefig(f"figures/figure_appendix_B_absorption.pdf")
