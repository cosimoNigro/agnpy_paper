import numpy as np
import astropy.units as u
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import CMB
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label
from pathlib import Path
from utils import time_function_call


# agnpy
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
blob.set_gamma_size(300)

cmb = CMB(z=blob.z)

ec_cmb = ExternalCompton(blob, cmb)

nu_ec = np.logspace(16, 29, 100) * u.Hz
sed_ec = time_function_call(ec_cmb.sed_flux, nu_ec)


# jetset
from jetset.jet_model import Jet

jet = Jet(
    name="ec_cmb",
    electron_distribution="bkn",
    electron_distribution_log_values=False,
    beaming_expr="bulk_theta",
)
jet.add_EC_component(["EC_CMB"])

# - blob
jet.set_par("N", val=blob.n_e_tot.value)
jet.set_par("p", val=blob.n_e.p1)
jet.set_par("p_1", val=blob.n_e.p2)
jet.set_par("gamma_break", val=blob.n_e.gamma_b)
jet.set_par("gmin", val=blob.n_e.gamma_min)
jet.set_par("gmax", val=blob.n_e.gamma_max)
jet.set_par("R", val=blob.R_b.value)
jet.set_par("B", val=blob.B.value)
jet.set_par("BulkFactor", val=blob.Gamma)
jet.set_par("theta", val=blob.theta_s.value)
jet.set_par("z_cosm", val=blob.z)

# - integration setup
jet.electron_distribution.update()
jet.set_gamma_grid_size(10000)
jet._blob.IC_adaptive_e_binning = True
jet.set_nu_grid(nu_ec[0].value, nu_ec[-1].value, len(nu_ec))
jet.set_external_field_transf("disk")

# - SED
jet.eval()

sed_ec_jetset = jet.spectral_components.EC_CMB.SED.nuFnu


# make figure 11
load_mpl_rc()
plt.rcParams["text.usetex"] = True

# gridspec plot setting
fig = plt.figure(tight_layout=True)
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1], figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)

# EC on CMB SEDs
ax1.loglog(nu_ec, sed_ec, ls="-", lw=2.1, color="crimson", label="agnpy")
ax1.loglog(nu_ec, sed_ec_jetset, ls="--", color="dodgerblue", label="jetset")
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best", fontsize=10)
ax1.set_title("EC on CMB, " + r"$z=1$")

# plot the deviation from the reference in the bottom panel
deviation_ref = sed_ec / sed_ec_jetset - 1

ax2.grid(False)
ax2.axhline(0, ls="-", color="darkgray")
ax2.axhline(0.2, ls="--", color="darkgray")
ax2.axhline(-0.2, ls="--", color="darkgray")
ax2.axhline(0.3, ls="-.", color="darkgray")
ax2.axhline(-0.3, ls="-.", color="darkgray")
ax2.set_ylim([-0.5, 0.5])
ax2.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax2.semilogx(
    nu_ec, deviation_ref, ls="--", lw=1.5, color="dodgerblue", label="jetset",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_xlabel(sed_x_label)
ax2.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")

Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_11.png")
fig.savefig(f"figures/figure_11.pdf")
