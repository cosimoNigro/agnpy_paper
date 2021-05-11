import numpy as np
import astropy.units as u
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.targets import RingDustTorus, PointSourceBehindJet
from agnpy.absorption import Absorption
from agnpy.utils.plot import load_mpl_rc


z = 0.859  # redshift of the source
L_disk = 2 * 1e46 * u.Unit("erg s-1")
# dust torus definition
T_dt = 1e3 * u.K
xi_dt = 0.1
dt = RingDustTorus(L_disk, xi_dt, T_dt)
# point source approximating the DT
ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)
# Absorptions
# - near the DT, aligned, to be compared with the reference
abs_dt_near = Absorption(dt, r=1.1e18 * u.cm, z=z)
# - misaligned case, to be checked against the point-source approximation
mu_s = np.cos(np.deg2rad(20))
abs_dt_far_mis = Absorption(dt, r=1e22 * u.cm, z=z, mu_s=mu_s)
abs_ps_dt_far_mis = Absorption(dt, r=1e22 * u.cm, z=z, mu_s=mu_s)

# reference SED, Figure 14 Finke Dermer
data_file_ref_abs = pkg_resources.resource_filename(
    "agnpy",
    "data/reference_taus/finke_2016/figure_14_left/tau_DT_r_1e1_R_Ly_alpha.txt",
)
data_ref = np.loadtxt(data_file_ref_abs, delimiter=",")
E_ref = data_ref[:, 0] * u.GeV
nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
tau_ref = 2 * data_ref[:, 1]  # correction to Finke's mistake in energy density formula

# enlarge the range of frequencies for the misaligned case
nu = np.logspace(25, 31) * u.Hz

# recompute agnpy absorption on the same frequency points of the reference
tau_dt_near = abs_dt_near.tau(nu_ref)
tau_dt_far_mis = abs_dt_far_mis.tau(nu)
tau_ps_dt_far_mis = abs_ps_dt_far_mis.tau(nu)


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
# optical depth near the DT
ax1.loglog(nu_ref, tau_dt_near, ls="-", lw=2, color="crimson", label="agnpy")
ax1.loglog(
    nu_ref, tau_ref, ls="--", lw=1.5, color="k", label="Fig. 14, Finke (2016)",
)
ax1.set_ylabel(r"$\tau_{\gamma\gamma}$")
ax1.legend(loc="best", fontsize=10)
ax1.set_title(
    "absorption on ring DT, "
    + r"$r=1.1 \times 10^{18}\,{\rm cm} < R_{\rm dt},\,\mu_s=0$"
)
ax1.set_ylim([1e-1, 1e3])
# optical depth far from the DT
ax2.loglog(
    nu, tau_dt_far_mis, ls="-", lw=2, color="crimson", label="agnpy, full calculation",
)
ax2.loglog(
    nu,
    tau_ps_dt_far_mis,
    ls="--",
    lw=1.5,
    color="k",
    label="agnpy, point-source approximation",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_title(
    "absorption on ring DT, " + r"$r=10^{22}\,{\rm cm} \gg R_{\rm dt},\,\mu_s \neq 0$"
)
ax2.set_ylim([1e-5, 1e-1])
# plot the deviation from the reference in the bottom panel
deviation_ref = tau_dt_near / tau_ref - 1
deviation_approx = tau_dt_far_mis / tau_ps_dt_far_mis - 1
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
    nu,
    deviation_approx,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$\tau_{\gamma\gamma, \rm agnpy}\,/\,\tau_{\gamma\gamma, \rm ref} - 1$",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
# save the figure
fig.savefig(f"figures/tau_dt_crosscheck.png")
fig.savefig(f"figures/tau_dt_crosscheck.pdf")
