# script to compare synchrotron and SSC against Dermer 2009 and jetset
import pkg_resources
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
from utils import reproduce_sed


# agnpy
spectrum_norm = 1e48 * u.Unit("erg")
spectrum_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10

blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

synch = Synchrotron(blob)
ssc = SynchrotronSelfCompton(blob)

nu_synch = np.logspace(9, 19, 100) * u.Hz
nu_ssc = np.logspace(14, 26, 100) * u.Hz

sed_synch = synch.sed_flux(nu_synch)
sed_ssc = ssc.sed_flux(nu_ssc)


# reproduce Figure 7.4 of Dermer 2009 with agnpy
# - synchrotron
data_synch_dermer = pkg_resources.resource_filename(
    "agnpy",
    "data/reference_seds/dermer_menon_2009/figure_7_4/synchrotron_gamma_max_1e5.txt",
)

synch_nu_range = [1e9, 5e18] * u.Hz
nu_synch_dermer, sed_synch_dermer, sed_synch_agnpy_dermer = reproduce_sed(
    data_synch_dermer, synch, synch_nu_range
)

# - SSC
data_ssc_dermer = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/dermer_menon_2009/figure_7_4/ssc_gamma_max_1e5.txt"
)

ssc_nu_range = [1e14, 1e26] * u.Hz
nu_ssc_dermer, sed_ssc_dermer, sed_ssc_agnpy_dermer = reproduce_sed(
    data_ssc_dermer, ssc, ssc_nu_range
)


# jetset
from jetset.jet_model import Jet

jet = Jet(
    name="ssc",
    electron_distribution="pl",
    electron_distribution_log_values=False,
    beaming_expr="bulk_theta",
)
jet.set_par("N", val=blob.n_e_tot.value)
jet.set_par("p", val=blob.n_e.p)
jet.set_par("gmin", val=blob.n_e.gamma_min)
jet.set_par("gmax", val=blob.n_e.gamma_max)
jet.set_par("R", val=blob.R_b.value)
jet.set_par("B", val=blob.B.value)
jet.set_par("BulkFactor", val=blob.Gamma)
jet.set_par("theta", val=blob.theta_s.value)
jet.set_par("z_cosm", val=blob.z)
# remove SSA
jet.spectral_components.Sync.state = "on"

# - synchrotron SED with jetset
jet.set_nu_grid(nu_synch[0].value, nu_synch[-1].value, len(nu_synch))
jet.eval()

nu_synch_jetset = jet.spectral_components.Sync.SED.nu
sed_synch_jetset = jet.spectral_components.Sync.SED.nuFnu

# - SSC SED with jetset
jet.set_nu_grid(nu_ssc[0].value, nu_ssc[-1].value, len(nu_ssc))
jet.eval()

nu_ssc_jetset = jet.spectral_components.SSC.SED.nu
sed_ssc_jetset = jet.spectral_components.SSC.SED.nuFnu


# make figure 7
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
    nu_synch, sed_synch, lw=2.1, ls="-", color="crimson",
)
ax1.loglog(
    nu_synch_dermer,
    sed_synch_agnpy_dermer,
    lw=2.1,
    ls="-",
    color="crimson",
    label="agnpy",
)
ax1.loglog(
    nu_synch_dermer,
    sed_synch_dermer,
    ls="--",
    color="k",
    label="Fig. 7.4, Dermer \& Menon (2009)",
)
ax1.loglog(
    nu_synch_jetset, sed_synch_jetset, ls="--", color="dodgerblue", label="jetset"
)
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best")
ax1.set_title("synchrotron")
ax1.set_ylim([1e-14, 1e-9])

# ssc SEDs
ax2.loglog(nu_ssc, sed_ssc, lw=2.1, ls="-", color="crimson")
ax2.loglog(
    nu_ssc_dermer, sed_ssc_agnpy_dermer, lw=2.1, ls="-", color="crimson", label="agnpy"
)
ax2.loglog(
    nu_ssc_dermer,
    sed_ssc_dermer,
    ls="--",
    color="k",
    label="Fig. 7.4, Dermer \& Menon (2009)",
)
ax2.loglog(nu_ssc_jetset, sed_ssc_jetset, ls="--", color="dodgerblue", label="jetset")
ax2.legend(loc="best")
ax2.set_title("synchrotron self-Compton")

# plot the deviation from the synchrotron references in the bottom panel
deviation_synch_dermer = sed_synch_agnpy_dermer / sed_synch_dermer - 1
deviation_synch_jetset = sed_synch / sed_synch_jetset - 1
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
    label="Fig. 7.4, Dermer \& Menon (2009)",
)
ax3.semilogx(
    nu_synch_jetset,
    deviation_synch_jetset,
    ls="--",
    color="dodgerblue",
    label="jetset",
)
ax3.legend(loc="best", fontsize=10)
ax3.set_xlabel(sed_x_label)
ax3.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")

# plot the deviation from the SSC reference in the bottom panel
deviation_ssc_dermer = sed_ssc_agnpy_dermer / sed_ssc_dermer - 1
deviation_ssc_jetset = sed_ssc / sed_ssc_jetset - 1
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
    label="Fig. 7.4, Dermer \& Menon (2009)",
)
ax4.semilogx(
    nu_ssc_jetset, deviation_ssc_jetset, ls="--", color="dodgerblue", label="jetset",
)
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(sed_x_label)
Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_7.png")
fig.savefig(f"figures/figure_7.pdf")
