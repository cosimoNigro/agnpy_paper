import numpy as np
import astropy.units as u
import astropy.constants as const
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SSDisk
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

# let us adopt the same disk parameters of Finke 2016
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6
R_out = 200
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
# point source approximating the DT
ps_disk_R_in = PointSourceBehindJet(L_disk, disk.epsilon(R_in))
ps_disk_R_out = PointSourceBehindJet(L_disk, disk.epsilon(R_out))

# define the EC
# - near the disk, to be compared with the reference
blob.set_gamma_size(300)
ec_disk_near = ExternalCompton(blob, disk, r=1e17 * u.cm)
# - far from the disk, to be compared with the point-source approximation
blob.set_gamma_size(400)
ec_disk_far = ExternalCompton(blob, disk, r=1e21 * u.cm)
blob.set_gamma_size(600)
ec_disk_ps_R_in = ExternalCompton(blob, ps_disk_R_in, r=1e21 * u.cm)
ec_disk_ps_R_out = ExternalCompton(blob, ps_disk_R_out, r=1e21 * u.cm)

# compute the SEDs
data_file_ref_disk = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/finke_2016/figure_8/ec_disk_r_1e17.txt"
)
# reference SED, Figure 8 Finke Dermer
data_ref = np.loadtxt(data_file_ref_disk, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
# make a denser frequency grid with intermediate points in log-scale
nu_denser = np.append(nu_ref, np.sqrt(nu_ref[1:] * nu_ref[:-1]))
nu = np.sort(nu_denser)
sed_ref = data_ref[:, 1] * u.Unit("erg cm-2 s-1")

# compute agnpy SEDs on the denser frequency grid
sed_agnpy_disk_near = time_function_call(ec_disk_near.sed_flux, nu)
sed_agnpy_disk_far = time_function_call(ec_disk_far.sed_flux, nu)
sed_agnpy_disk_R_in = time_function_call(ec_disk_ps_R_in.sed_flux, nu)
sed_agnpy_disk_R_out = time_function_call(ec_disk_ps_R_out.sed_flux, nu)


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
# SED close to the disk
ax1.loglog(nu, sed_agnpy_disk_near, ls="-", lw=2, color="crimson", label="agnpy")
ax1.loglog(
    nu_ref, sed_ref, ls="--", lw=1.5, color="k", label="Fig. 8, Finke (2016)",
)
ax1.set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
ax1.legend(loc="best", fontsize=10)
ax1.set_title("EC on Shakura Sunyaev disk, " + r"$r=10^{17}\,{\rm cm} < R_{\rm out}$")
# SED far from the disk
ax2.loglog(
    nu,
    sed_agnpy_disk_far,
    ls="-",
    lw=2,
    color="crimson",
    label="agnpy, full calculation",
)
ax2.loglog(
    nu,
    sed_agnpy_disk_R_in,
    ls="--",
    lw=1.5,
    color="k",
    label="agnpy, point-source approx., " + r"$\epsilon_0 = \epsilon_0(R_{\rm in})$",
)
ax2.loglog(
    nu,
    sed_agnpy_disk_R_out,
    ls=":",
    lw=1.5,
    color="k",
    label="agnpy, point-source approx., " + r"$\epsilon_0 = \epsilon_0(R_{\rm out})$",
)
# shade the area between the two SED of the point source approximations
ax2.fill_between(nu, sed_agnpy_disk_R_in, sed_agnpy_disk_R_out, color="silver")
ax2.legend(loc="best", fontsize=10)
ax2.set_title("EC on Shakura Sunyaev disk, " + r"$r=10^{21}\,{\rm cm} \gg R_{\rm out}$")
# plot the deviation from the reference in the bottom panel
# remove every other value from the SED to be compared with the reference
# as it has been calculated on the finer frequency grid
deviation_ref = sed_agnpy_disk_near[::2] / sed_ref - 1
deviation_approx_in = sed_agnpy_disk_far / sed_agnpy_disk_R_in - 1
deviation_approx_out = sed_agnpy_disk_far / sed_agnpy_disk_R_out - 1
ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.3, ls=":", color="darkgray")
ax3.axhline(-0.3, ls=":", color="darkgray")
ax3.set_ylim([-0.5, 0.5])
ax3.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax3.semilogx(
    nu_ref, deviation_ref, ls="--", lw=1.5, color="k", label="Fig. 8, Finke (2016)",
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
    deviation_approx_in,
    ls="--",
    lw=1.5,
    color="k",
    label="point-source approx., " + r"$\epsilon_0 = \epsilon_0(R_{\rm in})$",
)
ax4.semilogx(
    nu,
    deviation_approx_out,
    ls=":",
    lw=1.5,
    color="k",
    label="point-source approx., " + r"$\epsilon_0 = \epsilon_0(R_{\rm out})$",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_9.png")
fig.savefig(f"figures/figure_9.pdf")
