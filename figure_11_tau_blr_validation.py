import numpy as np
import astropy.units as u
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.targets import SphericalShellBLR, PointSourceBehindJet
from agnpy.absorption import Absorption
from agnpy.utils.plot import load_mpl_rc


z = 0.859  # redshift of the source
L_disk = 2 * 1e46 * u.Unit("erg s-1")
xi_line = 0.024
R_line = 1.1 * 1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
# point source with the same luminosity as the BLR
ps_blr = PointSourceBehindJet(blr.xi_line * L_disk, blr.epsilon_line)
# Absorptions
# - aligned case, to be checked against the reference
abs_in_blr = Absorption(blr, r=0.1 * R_line, z=z)
# - misaligned case, to be checked against the point-source approximation
mu_s = np.cos(np.deg2rad(20))
abs_out_blr_mis = Absorption(blr, r=1e3 * R_line, z=z, mu_s=mu_s)
abs_out_ps_blr_mis = Absorption(ps_blr, r=1e3 * R_line, z=z, mu_s=mu_s)

# reference SED, Figure 14 Finke Dermer
data_file_ref_abs = pkg_resources.resource_filename(
    "agnpy",
    "data/reference_taus/finke_2016/figure_14_left/tau_BLR_Ly_alpha_r_1e-1_R_Ly_alpha.txt",
)
data_ref = np.loadtxt(data_file_ref_abs, delimiter=",")
E_ref = data_ref[:, 0] * u.GeV
nu_ref = E_ref.to("Hz", equivalencies=u.spectral()) / (1 + z)
tau_ref = data_ref[:, 1]

# recompute agnpy absorption on the same frequency points of the reference
tau_in_blr = abs_in_blr.tau(nu_ref)
tau_out_blr_mis = abs_out_blr_mis.tau(nu_ref)
tau_out_ps_blr_mis = abs_out_ps_blr_mis.tau(nu_ref)


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
ax1.loglog(nu_ref, tau_in_blr, ls="-", lw=2, color="crimson", label="agnpy")
ax1.loglog(
    nu_ref, tau_ref, ls="--", lw=1.5, color="k", label="Fig. 14, Finke (2016)",
)
ax1.set_ylabel(r"$\tau_{\gamma\gamma}$")
ax1.legend(loc="best", fontsize=10)
ax1.set_title(
    "abs. on spherical shell BLR, "
    + r"$r=1.1 \times 10^{16}\,{\rm cm} < R_{\rm Ly \alpha},\,\mu_{\rm s}=0$"
)
ax1.set_ylim([1e-1, 1e3])
# SED outside the BLR
ax2.loglog(
    nu_ref,
    tau_out_blr_mis,
    ls="-",
    lw=2,
    color="crimson",
    label="agnpy, full calculation",
)
ax2.loglog(
    nu_ref,
    tau_out_ps_blr_mis,
    ls="--",
    lw=1.5,
    color="k",
    label="agnpy, point-source approximation",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_title(
    "abs. on spherical shell BLR, "
    + r"$r=1.1 \times 10^{20}\,{\rm cm} \gg R_{\rm Ly \alpha},\,\mu_{\rm s} \neq 0$"
)
ax2.set_ylim([1e-6, 1e-2])
# plot the deviation from the reference in the bottom panel
deviation_ref = tau_in_blr / tau_ref - 1
deviation_approx = tau_out_blr_mis / tau_out_ps_blr_mis - 1
ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.3, ls=":", color="darkgray")
ax3.axhline(-0.3, ls=":", color="darkgray")
ax3.set_ylim([-0.5, 0.5])
ax3.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax3.semilogx(
    nu_ref,
    deviation_ref,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$\tau_{\gamma\gamma, \rm agnpy}\,/\,\tau_{\gamma\gamma, \rm ref} - 1$",
)
ax3.legend(loc="best", fontsize=10)
ax3.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
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
    nu_ref,
    deviation_approx,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$\tau_{\gamma\gamma, \rm agnpy}\,/\,\tau_{\gamma\gamma, \rm ref} - 1$",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
# save the figure
fig.savefig(f"figures/figure_11.png")
fig.savefig(f"figures/figure_11.pdf")
