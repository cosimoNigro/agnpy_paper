import numpy as np
import astropy.units as u
import pkg_resources
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SphericalShellBLR, RingDustTorus
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt

load_mpl_rc()

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
blob.set_gamma_size(700)

L_disk = 2 * 1e46 * u.Unit("erg s-1")

# check DT for very large distance
# DT definition
# dust torus definition
T_dt = 1e3 * u.K
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, T_dt)
# point source approximating the DT
ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)
# EC
ec_dt = ExternalCompton(blob, dt, r=1e21 * u.cm)
ec_ps_dt = ExternalCompton(blob, ps_dt, r=1e21 * u.cm)

# plot SEDs
data_file_ref_blr = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/finke_2016/figure_11/ec_dt_r_1e21.txt"
)
# reference SED, Figure 11 Finke Dermer
data_ref = np.loadtxt(data_file_ref_blr, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
sed_ref = data_ref[:, 1] * u.Unit("erg cm-2 s-1")

# recompute agnpy SEDs on the same frequency points of the reference
sed_agnpy_dt = ec_dt.sed_flux(nu_ref)
sed_agnpy_ps_dt = ec_ps_dt.sed_flux(nu_ref)


# figure
load_mpl_rc()
fig, ax = plt.subplots(
    2,
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08, "top": 0.95},
    figsize=(8, 6),
)
fig.set_tight_layout(False)
ax[0].loglog(
    nu_ref, sed_agnpy_dt, ls="-", lw=2, color="crimson", label="agnpy, EC on DT"
)
ax[0].loglog(
    nu_ref,
    sed_agnpy_ps_dt,
    ls="-",
    lw=1.5,
    color="dodgerblue",
    label="agnpy, EC on point-source",
)
ax[0].loglog(
    nu_ref, sed_ref, ls="--", lw=1.5, color="k", label="Figure 11, Finke (2016)",
)
ax[0].set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
ax[0].legend(loc="best", fontsize=10)
# ax[0].set_ylim([1e-13, 1e-9])
# plot the deviation in the bottom panel
deviation_full = sed_agnpy_dt / sed_ref - 1
deviation_approx = sed_agnpy_ps_dt / sed_ref - 1
ax[1].grid(False)
ax[1].axhline(0, ls="-", color="darkgray")
ax[1].axhline(0.2, ls="--", color="darkgray")
ax[1].axhline(-0.2, ls="--", color="darkgray")
ax[1].axhline(0.3, ls=":", color="darkgray")
ax[1].axhline(-0.3, ls=":", color="darkgray")
ax[1].set_ylim([-0.5, 0.5])
ax[1].set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax[1].semilogx(
    nu_ref,
    deviation_full,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm ref} - 1$",
)
ax[1].semilogx(
    nu_ref,
    deviation_approx,
    ls="--",
    lw=1.5,
    color="dodgerblue",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm ref} - 1$",
)
ax[1].set_xlabel(r"$\nu\,/\,{\rm Hz}$")
ax[1].legend(loc="best", fontsize=10)
plt.show()
fig.savefig(f"figures/ec_dt_crosscheck.pdf")
