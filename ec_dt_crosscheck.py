import numpy as np
import astropy.units as u
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, RingDustTorus
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc


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

# dust torus definition
T_dt = 1e3 * u.K
xi_dt = 0.1
dt = RingDustTorus(L_disk, xi_dt, T_dt)
# point source approximating the DT
ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)
# EC
# - near the DT, to be compared with the reference
blob.set_gamma_size(500)
ec_dt_near = ExternalCompton(blob, dt, r=1e18 * u.cm)
# - far from the DT, to be compared with the point-source approximation
blob.set_gamma_size(300)
ec_dt_far = ExternalCompton(blob, dt, r=1e22 * u.cm)
blob.set_gamma_size(600)
ec_ps_dt = ExternalCompton(blob, ps_dt, r=1e22 * u.cm)

# reference SED, Figure 11 Finke Dermer
data_file_ref_dt_near = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/finke_2016/figure_11/ec_dt_r_1e18.txt"
)
data_ref = np.loadtxt(data_file_ref_dt_near, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
sed_ref = data_ref[:, 1] * u.Unit("erg cm-2 s-1")

# recompute agnpy SEDs on the same frequency points of the reference
sed_agnpy_dt_near = ec_dt_near.sed_flux(nu_ref)
sed_agnpy_dt_far = ec_dt_far.sed_flux(nu_ref)
sed_agnpy_ps_dt = ec_ps_dt.sed_flux(nu_ref)


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
ax1.loglog(nu_ref, sed_agnpy_dt_near, ls="-", lw=2, color="crimson", label="agnpy")
ax1.loglog(
    nu_ref, sed_ref, ls="--", lw=1.5, color="k", label="Fig. 11, Finke (2016)",
)
ax1.set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
ax1.legend(loc="best", fontsize=10)
ax1.set_title("EC on ring DT, " + r"$r=10^{18}\,{\rm cm} < R_{\rm DT}$")
# SED outside the BLR
ax2.loglog(
    nu_ref,
    sed_agnpy_dt_far,
    ls="-",
    lw=2,
    color="crimson",
    label="agnpy, full calculation",
)
ax2.loglog(
    nu_ref,
    sed_agnpy_ps_dt,
    ls="--",
    lw=1.5,
    color="k",
    label="agnpy, point-source approximation",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_title("EC on ring DT, " + r"$r=10^{22}\,{\rm cm} \gg R_{\rm DT}$")
# plot the deviation from the reference in the bottom panel
deviation_ref = sed_agnpy_dt_near / sed_ref - 1
deviation_approx = sed_agnpy_dt_far / sed_agnpy_ps_dt - 1
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
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm ref} - 1$",
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
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm approx} - 1$",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
# save the figure
fig.savefig(f"figures/ec_dt_crosscheck.png")
fig.savefig(f"figures/ec_dt_crosscheck.pdf")
