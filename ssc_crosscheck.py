import numpy as np
import pkg_resources
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt

spectrum_norm = 1e48 * u.Unit("erg")
pwl_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5,},
}
# blob parameters
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
# define emission region and radiative process
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, pwl_dict)
ssc = SynchrotronSelfCompton(blob)

data_file_jetset = pkg_resources.resource_filename(
    "agnpy", "data/sampled_seds/ssc_pwl_jetset_1.1.2.txt"
)
data_file_ref = pkg_resources.resource_filename(
    "agnpy", "data/sampled_seds/ssc_figure_7_4_dermer_menon_2009.txt"
)

# jetset SED
data_jetset = np.loadtxt(data_file_jetset, delimiter=",")
nu_jetset = data_jetset[:, 0] * u.Hz
sed_jetset = data_jetset[:, 1] * u.Unit("erg cm-2 s-1")

# reference SED, Figure 7.4 Dermer
data_ref = np.loadtxt(data_file_ref, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
sed_ref = data_ref[:, 1] * u.Unit("erg cm-2 s-1")

# agnpy SED recomputed on jetset frequencies
sed_agnpy_nu_jetset = ssc.sed_flux(nu_jetset)
# agnpy SED recomputed on reference frequencies
sed_agnpy_nu_ref = ssc.sed_flux(nu_ref)

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
    nu_jetset, sed_agnpy_nu_jetset, ls="-", lw=2, color="crimson", label="agnpy"
)
ax[0].loglog(nu_jetset, sed_jetset, ls="--", lw=1.5, color="k", label="jetset")
ax[0].loglog(
    nu_ref,
    sed_ref,
    ls="--",
    lw=1.5,
    color="dodgerblue",
    label="Figure 7.4, Dermer & Menon (2009)",
)
ax[0].set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
ax[0].legend(loc="best", fontsize=10)
ax[0].set_ylim([1e-13, 1e-9])
# plot the deviation in the bottom panel
deviation_jetset = 1 - sed_agnpy_nu_jetset / sed_jetset
deviation_ref = 1 - sed_agnpy_nu_ref / sed_ref
ax[1].grid(False)
ax[1].axhline(0, ls="-", color="darkgray")
ax[1].axhline(0.2, ls="--", color="darkgray")
ax[1].axhline(-0.2, ls="--", color="darkgray")
ax[1].axhline(0.3, ls=":", color="darkgray")
ax[1].axhline(-0.3, ls=":", color="darkgray")
ax[1].set_ylim([-0.5, 0.5])
ax[1].set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax[1].semilogx(
    nu_jetset,
    deviation_jetset,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$1 - \nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm jetset}$",
)
ax[1].semilogx(
    nu_ref,
    deviation_ref,
    ls="--",
    lw=1.5,
    color="dodgerblue",
    label=r"$1 - \nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm reference}$",
)
ax[1].set_xlabel(r"$\nu\,/\,{\rm Hz}$")
ax[1].legend(loc="best", fontsize=10)
# plt.show()
fig.savefig(f"figures/ssc_crosscheck.pdf")
