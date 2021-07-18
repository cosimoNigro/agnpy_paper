import numpy as np
import astropy.units as u
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SphericalShellBLR
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc
from pathlib import Path
from utils import time_function_call

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

L_disk = 2 * 1e46 * u.Unit("erg s-1")

# check BLR for very large distance
# BLR definition
xi_line = 0.024
R_line = 1.1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
# point source approximating the BLR
ps_blr = PointSourceBehindJet(blr.xi_line * L_disk, blr.epsilon_line)
# EC
# - inside the BLR, to be compared with the reference
blob.set_gamma_size(350)
ec_blr_in = ExternalCompton(blob, blr, r=1e16 * u.cm)
# - outside the BLR, to be compared with the point-source approximation
blob.set_gamma_size(350)
ec_blr_out = ExternalCompton(blob, blr, r=1e20 * u.cm)
blob.set_gamma_size(700)
ec_ps_blr = ExternalCompton(blob, ps_blr, r=1e20 * u.cm)

# plot SEDs
data_file_ref_blr_in = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/finke_2016/figure_10/ec_blr_r_1e16.txt"
)
# reference SED, Figure 10 Finke Dermer
data_ref = np.loadtxt(data_file_ref_blr_in, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
# make a denser frequency grid with intermediate points in log-scale
nu_denser = np.append(nu_ref, np.sqrt(nu_ref[1:] * nu_ref[:-1]))
nu = np.sort(nu_denser)
sed_ref = data_ref[:, 1] * u.Unit("erg cm-2 s-1")

# compute agnpy SEDs on the denser frequency grid
sed_agnpy_blr_in = time_function_call(ec_blr_in.sed_flux, nu)
sed_agnpy_blr_out = time_function_call(ec_blr_out.sed_flux, nu)
sed_agnpy_ps_blr = time_function_call(ec_ps_blr.sed_flux, nu)


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
# SED inside the BLR
ax1.loglog(nu, sed_agnpy_blr_in, ls="-", lw=2, color="crimson", label="agnpy")
ax1.loglog(
    nu_ref, sed_ref, ls="--", lw=1.5, color="k", label="Fig. 10, Finke (2016)",
)
ax1.set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
ax1.legend(loc="best", fontsize=10)
ax1.set_title(
    "EC on spherical shell BLR, "
    + r"$r=1.1 \times 10^{16}\,{\rm cm} < R_{\rm Ly \alpha}$"
)
# SED outside the BLR
ax2.loglog(
    nu,
    sed_agnpy_blr_out,
    ls="-",
    lw=2,
    color="crimson",
    label="agnpy, full calculation",
)
ax2.loglog(
    nu,
    sed_agnpy_ps_blr,
    ls="--",
    lw=1.5,
    color="k",
    label="agnpy, point-source approximation",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_title(
    "EC on spherical shell BLR, "
    + r"$r=1.1 \times 10^{20}\,{\rm cm} \gg R_{\rm Ly \alpha}$"
)
# plot the deviation from the reference in the bottom panel
# remove every other value from the SED to be compared with the reference
# as it has been calculated on the finer frequency grid
deviation_ref = sed_agnpy_blr_in[::2] / sed_ref - 1
deviation_approx = sed_agnpy_blr_out / sed_agnpy_ps_blr - 1
ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.3, ls=":", color="darkgray")
ax3.axhline(-0.3, ls=":", color="darkgray")
ax3.set_ylim([-0.5, 0.5])
ax3.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax3.semilogx(
    nu_ref, deviation_ref, ls="--", lw=1.5, color="k", label="Fig. 10, Finke (2016)",
)
ax3.legend(loc="best", fontsize=10)
ax3.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
ax3.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")
# plot the deviation from the point like approximation in the bottom panel
ax4.grid(False)
ax4.axhline(0, ls="-", color="darkgray")
ax4.axhline(0.2, ls="--", color="darkgray")
ax4.axhline(-0.2, ls="--", color="darkgray")
ax4.axhline(0.3, ls=":", color="darkgray")
ax4.axhline(-0.3, ls=":", color="darkgray")
ax4.set_ylim([-0.5, 0.5])
ax4.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax4.semilogx(
    nu,
    deviation_approx,
    ls="--",
    lw=1.5,
    color="k",
    label="point-source approximation",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_10.png")
fig.savefig(f"figures/figure_10.pdf")
