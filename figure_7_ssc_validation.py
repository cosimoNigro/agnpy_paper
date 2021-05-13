import numpy as np
import pkg_resources
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def reproduce_sed(dataset, process, nu_range):
    """function to reproduce the SED data in a given reference dataset"""
    # reference SED
    sed_data = np.loadtxt(dataset, delimiter=",")
    nu_ref = sed_data[:, 0] * u.Hz
    # apply the comparison range
    comparison = (nu_ref >= nu_range[0]) * (nu_ref <= nu_range[-1])
    nu_ref = nu_ref[comparison]
    sed_ref = sed_data[:, 1][comparison] * u.Unit("erg cm-2 s-1")
    # compute the sed with agnpy on the same frequencies
    sed_agnpy = process.sed_flux(nu_ref)
    return nu_ref, sed_ref, sed_agnpy


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
ssc = SynchrotronSelfCompton(blob)

# reference datasets
data_synch_jetset = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/jetset/synch_pwl_jetset_1.1.2.txt"
)
data_synch_dermer = pkg_resources.resource_filename(
    "agnpy",
    "data/reference_seds/dermer_menon_2009/figure_7_4/synchrotron_gamma_max_1e5.txt",
)
data_ssc_jetset = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/jetset/ssc_pwl_jetset_1.1.2.txt"
)
data_ssc_dermer = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/dermer_menon_2009/figure_7_4/ssc_gamma_max_1e5.txt"
)

# synch SEDs
synch_nu_range = [1e9, 5e18] * u.Hz
nu_synch_jetset, sed_synch_agnpy_jetset, sed_synch_jetset = reproduce_sed(
    data_synch_jetset, synch, synch_nu_range
)
nu_synch_dermer, sed_synch_agnpy_dermer, sed_synch_dermer = reproduce_sed(
    data_synch_dermer, synch, synch_nu_range
)
# ssc SEDs
ssc_nu_range = [1e14, 1e26] * u.Hz
nu_ssc_jetset, sed_ssc_agnpy_jetset, sed_ssc_jetset = reproduce_sed(
    data_ssc_jetset, ssc, ssc_nu_range
)
nu_ssc_dermer, sed_ssc_agnpy_dermer, sed_ssc_dermer = reproduce_sed(
    data_ssc_dermer, ssc, ssc_nu_range
)

# figure
load_mpl_rc()
plt.rcParams["text.usetex"] = True
# gridspec plot setting
fig = plt.figure(figsize=(12, 6), tight_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[2, 1], figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
ax3 = fig.add_subplot(spec[1, 0], sharex=ax1)
ax4 = fig.add_subplot(spec[1, 1], sharex=ax2, sharey=ax3)
# synch SEDs
ax1.loglog(
    nu_synch_dermer, sed_synch_agnpy_dermer, ls="-", color="crimson", label="agnpy"
)
ax1.loglog(
    nu_synch_dermer,
    sed_synch_dermer,
    ls="--",
    color="k",
    label="Fig. 7.4, Dermer (2009)",
)
ax1.loglog(
    nu_synch_jetset, sed_synch_jetset, ls="--", color="dodgerblue", label="jetset"
)
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best")
ax1.set_title("synchrotron")
ax1.set_ylim([1e-14, 1e-9])
# ssc SEDs
ax2.loglog(nu_ssc_dermer, sed_ssc_agnpy_dermer, ls="-", color="crimson", label="agnpy")
ax2.loglog(
    nu_ssc_dermer, sed_ssc_dermer, ls="--", color="k", label="Fig. 7.4, Dermer (2009)"
)
ax2.loglog(nu_ssc_jetset, sed_ssc_jetset, ls="--", color="dodgerblue", label="jetset")
ax2.legend(loc="best")
ax2.set_title("synchrotron self-Compton")
# plot the deviation from the synchrotron reference in the bottom panel
deviation_synch_dermer = sed_synch_agnpy_dermer / sed_synch_dermer - 1
deviation_synch_jetset = sed_synch_agnpy_jetset / sed_synch_jetset - 1
ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.3, ls=":", color="darkgray")
ax3.axhline(-0.3, ls=":", color="darkgray")
ax3.set_ylim([-0.5, 0.5])
ax3.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax3.semilogx(
    nu_synch_dermer,
    deviation_synch_dermer,
    ls="--",
    color="k",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm ref} - 1$",
)
ax3.semilogx(
    nu_synch_jetset,
    deviation_synch_jetset,
    ls="--",
    color="dodgerblue",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm jetset} - 1$",
)
ax3.legend(loc="best", fontsize=10)
ax3.set_xlabel(sed_x_label)
# plot the deviation from the synchrotron reference in the bottom panel
deviation_ssc_dermer = sed_ssc_agnpy_dermer / sed_ssc_dermer - 1
deviation_ssc_jetset = sed_ssc_agnpy_jetset / sed_ssc_jetset - 1
ax4.grid(False)
ax4.axhline(0, ls="-", color="darkgray")
ax4.axhline(0.2, ls="--", color="darkgray")
ax4.axhline(-0.2, ls="--", color="darkgray")
ax4.axhline(0.3, ls=":", color="darkgray")
ax4.axhline(-0.3, ls=":", color="darkgray")
ax4.set_ylim([-0.5, 0.5])
ax4.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax4.semilogx(
    nu_ssc_dermer,
    deviation_ssc_dermer,
    ls="--",
    lw=1.5,
    color="k",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm ref} - 1$",
)
ax4.semilogx(
    nu_ssc_jetset,
    deviation_ssc_jetset,
    ls="--",
    color="dodgerblue",
    label=r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm jetset} - 1$",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(sed_x_label)
# save the figure
fig.savefig(f"figures/figure_7.png")
fig.savefig(f"figures/figure_7.pdf")
