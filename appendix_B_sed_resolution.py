# script to study the effect of the integration grid on the final SED resolution
import numpy as np
import astropy.units as u
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import RingDustTorus
from agnpy.synchrotron import Synchrotron
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
from pathlib import Path

# blob
spectrum_norm = 6e42 * u.erg
parameters = {
    "p1": 2.0,
    "p2": 3.5,
    "gamma_b": 1e4,
    "gamma_min": 20,
    "gamma_max": 5e7,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
# dust torus definition
L_disk = 2 * 1e46 * u.Unit("erg s-1")
T_dt = 1e3 * u.K
xi_dt = 0.1
dt = RingDustTorus(L_disk, xi_dt, T_dt)

# different resolution
n_gamma_list = [100, 200, 300, 400, 500]
n_phi_list = [50, 100, 200, 300, 400]
colors = ["crimson", "dodgerblue", "goldenrod", "lightseagreen", "k"]
# same frequency grid
nu = np.logspace(15, 29, 100) * u.Hz

# show the effect of changing the Lorentz factor and frequency grid
seds_variable_gamma = []
labels_variable_gamma = []
for n_gamma in n_gamma_list:
    blob.set_gamma_size(n_gamma)
    ec = ExternalCompton(blob, dt, r=1e19 * u.cm)
    sed = ec.sed_flux(nu)
    label = r"$n_{\gamma}=$" + f"{n_gamma}, " + r"$n_{\nu}=100,\,n_{\phi}=50$"
    seds_variable_gamma.append(sed)
    labels_variable_gamma.append(label)

# show the effect of changing the azimuth angle grid
seds_variable_phi = []
labels_variable_phi = []
for n_phi in n_phi_list:
    blob.set_gamma_size(500)
    ec = ExternalCompton(blob, dt, r=1e19 * u.cm)
    ec.set_phi(n_phi)
    sed = ec.sed_flux(nu)
    label = r"$n_{\gamma}=500,\,n_{\nu}=100,\,n_{\phi}=$" + f"{n_phi}"
    seds_variable_phi.append(sed)
    labels_variable_phi.append(label)

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
    seds_variable_gamma[:-1], labels_variable_gamma[:-1], colors[:-1]
):
    ax1.loglog(nu, sed, ls="-", color=color, label=label)
# plot the last one as reference
ax1.loglog(
    nu,
    seds_variable_gamma[-1],
    ls="--",
    color=colors[-1],
    label=labels_variable_gamma[-1],
)
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best", fontsize=10)
ax1.set_title("EC on ring DT, " + r"$r=10^{19}\,{\rm cm}$")
# changing azimuth angle grid
for sed, label, color in zip(
    seds_variable_phi[:-1], labels_variable_phi[:-1], colors[:-1]
):
    ax2.loglog(nu, sed, ls="-", color=color, label=label)
# plot the last one as reference
ax2.loglog(
    nu, seds_variable_phi[-1], ls="--", color=colors[-1], label=labels_variable_phi[-1]
)
ax2.set_ylabel(sed_y_label)
ax2.legend(loc="best", fontsize=10)
ax2.set_title("EC on ring DT, " + r"$r=10^{19}\,{\rm cm}$")
# plot the deviation from the denser SED in the bottom panel
for sed, label, color in zip(
    seds_variable_gamma[:-1], labels_variable_gamma[:-1], colors[:-1]
):
    deviation = (sed / seds_variable_gamma[-1]) - 1
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
    seds_variable_phi[:-1], labels_variable_phi[:-1], colors[:-1]
):
    deviation = (sed / seds_variable_phi[-1]) - 1
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
fig.savefig(f"figures/figure_appendix_B.png")
fig.savefig(f"figures/figure_appendix_B.pdf")
