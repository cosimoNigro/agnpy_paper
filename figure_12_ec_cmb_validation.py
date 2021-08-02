import numpy as np
import astropy.units as u
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import CMB
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
from pathlib import Path
from utils import reproduce_sed

# emission region
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
blob.set_gamma_size(300)
# target
cmb = CMB(z=blob.z)
# EC
ec_cmb = ExternalCompton(blob, cmb)

# generate another to be compared against the one produced by jetset
data_ec_cmb = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/jetset/data/ec_cmb_bpwl_jetset_1.1.2.txt"
)
ec_cmb_nu_range = [1e15, 1e29] * u.Hz
nu_jetset, sed_ec_jetset, sed_ec_agnpy = reproduce_sed(
    data_ec_cmb, ec_cmb, ec_cmb_nu_range
)

# figure
load_mpl_rc()
plt.rcParams["text.usetex"] = True
# gridspec plot setting
fig = plt.figure(tight_layout=True)
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1], figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
# EC on CMB SEDs
ax1.loglog(nu_jetset, sed_ec_agnpy, ls="-", lw=2, color="crimson", label="agnpy")
ax1.loglog(
    nu_jetset, sed_ec_jetset, ls="--", lw=1.5, color="k", label="jetset",
)
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best", fontsize=10)
ax1.set_title("EC on CMB, " + r"$z=1$")
# plot the deviation from the reference in the bottom panel
deviation_ref = sed_ec_agnpy / sed_ec_jetset - 1
ax2.grid(False)
ax2.axhline(0, ls="-", color="darkgray")
ax2.axhline(0.2, ls="--", color="darkgray")
ax2.axhline(-0.2, ls="--", color="darkgray")
ax2.axhline(0.3, ls=":", color="darkgray")
ax2.axhline(-0.3, ls=":", color="darkgray")
ax2.set_ylim([-0.5, 0.5])
ax2.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax2.semilogx(
    nu_jetset, deviation_ref, ls="--", lw=1.5, color="k", label="jetset",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_xlabel(sed_x_label)
ax2.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_12.png")
fig.savefig(f"figures/figure_12.pdf")
