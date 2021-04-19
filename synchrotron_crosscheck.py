import numpy as np
import pkg_resources
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt


# emission region
spectrum_norm = 1e48 * u.Unit("erg")
pwl_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
# blob parameters
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
# define emission region and radiative process
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, pwl_dict)
synch = Synchrotron(blob)

data_file_jetset = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/jetset/synch_pwl_jetset_1.1.2.txt"
)
data_file_ref = pkg_resources.resource_filename(
    "agnpy",
    "data/reference_seds/dermer_menon_2009/figure_7_4/synchrotron_gamma_max_1e5.txt",
)

# comparison range for nu
nu_comparison_range = [1e9, 5e18] * u.Hz

# jetset SED
data_jetset = np.loadtxt(data_file_jetset, delimiter=",")
nu_jetset = data_jetset[:, 0] * u.Hz
# apply the comparison range
comparison = (nu_jetset >= nu_comparison_range[0]) * (
    nu_jetset <= nu_comparison_range[-1]
)
nu_jetset = nu_jetset[comparison]
sed_jetset = data_jetset[:, 1][comparison] * u.Unit("erg cm-2 s-1")

# reference SED, Figure 7.4 Dermer
data_ref = np.loadtxt(data_file_ref, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
# apply the comparison range
comparison = (nu_ref >= nu_comparison_range[0]) * (nu_ref <= nu_comparison_range[-1])
nu_ref = nu_ref[comparison]
sed_ref = data_ref[:, 1][comparison] * u.Unit("erg cm-2 s-1")

# agnpy SED recomputed on jetset frequencies
sed_agnpy_nu_jetset = synch.sed_flux(nu_jetset)
# agnpy SED recomputed on reference frequencies
sed_agnpy_nu_ref = synch.sed_flux(nu_ref)

# figure
load_mpl_rc()
fig, ax = plt.subplots(
    2,
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08, "top": 0.94},
    figsize=(8, 6),
)
fig.set_tight_layout(False)
ax[0].loglog(nu_ref, sed_agnpy_nu_ref, ls="-", lw=2, color="crimson", label="agnpy")
ax[0].loglog(
    nu_ref,
    sed_ref,
    ls="--",
    lw=1.5,
    color="k",
    label="Figure 7.4, Dermer & Menon (2009)",
)
ax[0].loglog(nu_jetset, sed_jetset, ls="--", lw=1.5, color="dodgerblue", label="jetset")
ax[0].set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
ax[0].legend(loc="best", fontsize=10)
ax[0].set_ylim([1e-13, 1e-9])
# plot the deviation in the bottom panel
deviation_jetset = sed_agnpy_nu_jetset / sed_jetset - 1
deviation_ref = sed_agnpy_nu_ref / sed_ref - 1
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
    deviation_ref,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm ref} - 1$",
)
ax[1].semilogx(
    nu_jetset,
    deviation_jetset,
    ls="--",
    lw=1.5,
    color="dodgerblue",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm jetset} - 1$",
)
ax[1].set_xlabel(r"$\nu\,/\,{\rm Hz}$")
ax[1].legend(loc="best", fontsize=10)
fig.suptitle("Synchrotron")
plt.show()
fig.savefig(f"figures/synchrotron_crosscheck.png")
fig.savefig(f"figures/synchrotron_crosscheck.pdf")
