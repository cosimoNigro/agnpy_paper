# script to compare the EC on BLR against Finke 2016 and jetset
import numpy as np
import astropy.units as u
import pkg_resources
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SphericalShellBLR
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

# BLR parameters of Finke 2016
L_disk = 2 * 1e46 * u.Unit("erg s-1")
xi_line = 0.024
R_line = 1.1e17 * u.cm

blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

# point sources approximating the BLR at very large distances
ps_blr = PointSourceBehindJet(blr.xi_line * L_disk, blr.epsilon_line)

# EC definition
# - inside the BLR, to be compared with the references
r_in = 1e16 * u.cm
blob.set_gamma_size(350)
ec_in = ExternalCompton(blob, blr, r=r_in)

# - out of the BLR, to be compared with the point-source approximation
r_out = 1e20 * u.cm
blob.set_gamma_size(350)
ec_out = ExternalCompton(blob, blr, r=r_out)
blob.set_gamma_size(700)
ec_ps = ExternalCompton(blob, ps_blr, r=r_out)

nu_ec = np.logspace(16, 28, 100) * u.Hz
sed_ec_in = ec_in.sed_flux(nu_ec)
sed_ec_out = ec_out.sed_flux(nu_ec)
sed_ec_ps = ec_ps.sed_flux(nu_ec)


# reproduce Figure 10 of Finke 2016 with agnpy
data_file_ref_blr = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/finke_2016/figure_10/ec_blr_r_1e16.txt"
)
# reference SED, Figure 10 Finke Dermer
data_ref = np.loadtxt(data_file_ref_blr, delimiter=",")
nu_ref = data_ref[:, 0] * u.Hz
# make a denser frequency grid with intermediate points in log-scale
nu_denser = np.append(nu_ref, np.sqrt(nu_ref[1:] * nu_ref[:-1]))
nu = np.sort(nu_denser)
sed_ref = data_ref[:, 1] * u.Unit("erg cm-2 s-1")

# compute agnpy SEDs on the denser frequency grid
sed_ec_in_finke = time_function_call(ec_in.sed_flux, nu)


# jetset
from jetset.jet_model import Jet

jet = Jet(
    name="ec_blr",
    electron_distribution="bkn",
    electron_distribution_log_values=False,
    beaming_expr="bulk_theta",
)
jet.add_EC_component(["EC_BLR"], disk_type="BB")

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

# - BLR
jet.set_par("L_Disk", val=L_disk.value)
jet.set_par("tau_BLR", val=blr.xi_line)
jet.set_par("R_BLR_in", val=0.8 * blr.R_line.value)  # very thin BLR
jet.set_par("R_BLR_out", val=blr.R_line.value)

# - integration setup
jet.electron_distribution.update()
jet.set_gamma_grid_size(10000)
jet._blob.IC_adaptive_e_binning = True
jet._blob.theta_n_int = 500
jet.set_nu_grid(nu_ec[0].value, nu_ec[-1].value, len(nu_ec))

# - SED inside the BLR
jet.set_par("R_H", val=r_in.to_value("cm"))
jet.set_external_field_transf("disk")
# fixes by Andrea to reproduce Finke's approach
jet._blob.R_H_scale_factor = max(50, r_in.to_value("cm") / blr.R_line.to_value("cm"))
jet._blob.R_H_scale_factor = min(50, jet._blob.R_H_scale_factor)
jet._blob.R_H_lim = 0.5
jet._blob.theta_lim = 5

jet.eval()

sed_ec_in_jetset = jet.spectral_components.EC_BLR.SED.nuFnu

# - SED outside and very far from the BLR
jet.set_par("R_H", val=r_out.to_value("cm"))
jet.set_external_field_transf("disk")
# fixes by Andrea to reproduce Finke's approach
jet._blob.R_H_scale_factor = max(50, r_in.to_value("cm") / blr.R_line.to_value("cm"))
jet._blob.R_H_scale_factor = min(50, jet._blob.R_H_scale_factor)
jet._blob.R_H_lim = 0.5
jet._blob.theta_lim = 5

jet.eval()

sed_ec_out_jetset = jet.spectral_components.EC_BLR.SED.nuFnu


# figure 9
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
# ax1.loglog(nu_ec, sed_ec_in, ls="-", lw=2.1, color="crimson")
ax1.loglog(nu, sed_ec_in_finke, ls="-", lw=2.1, color="crimson", label="agnpy")
ax1.loglog(nu_ref, sed_ref, ls="--", color="k", label="Fig. 10, Finke (2016)")
ax1.loglog(nu_ec, sed_ec_in_jetset, ls="--", color="dodgerblue", label="jetset")
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best", fontsize=10)
ax1.set_title(
    "EC on spherical shell BLR, "
    + r"$r=1 \times 10^{16}\,{\rm cm} < R_{\rm Ly \alpha}$"
)

# SED outside the BLR
ax2.loglog(
    nu_ec, sed_ec_out, ls="-", lw=2.1, color="crimson", label="agnpy, full calculation",
)
ax2.loglog(
    nu_ec, sed_ec_ps, ls="--", color="k", label="agnpy, point-source approximation",
)
ax2.loglog(
    nu_ec, sed_ec_out_jetset, ls="--", color="dodgerblue", label="jetset",
)
ax2.legend(loc="best", fontsize=10)
ax2.set_title(
    "EC on spherical shell BLR, "
    + r"$r=1 \times 10^{20}\,{\rm cm} \gg R_{\rm Ly \alpha}$"
)

# plot the deviation from the references in the bottom panel
# remove every other value from the SED to be compared with the reference
# as it has been calculated on the finer frequency grid
deviation_ref = sed_ec_in_finke[::2] / sed_ref - 1
deviation_jetset_in = sed_ec_in / sed_ec_in_jetset - 1

ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.3, ls="-.", color="darkgray")
ax3.axhline(-0.3, ls="-.", color="darkgray")
ax3.axhline(0.5, ls=":", color="darkgray")
ax3.axhline(-0.5, ls=":", color="darkgray")
ax3.set_ylim([-1.1, 1.1])
ax3.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax3.semilogx(
    nu_ref, deviation_ref, ls="--", color="k", label="Fig. 10, Finke (2016)",
)
ax3.semilogx(
    nu_ec, deviation_jetset_in, ls="--", color="dodgerblue", label="jetset",
)
ax3.legend(loc="best", fontsize=10)
ax3.set_xlabel(sed_x_label)
ax3.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")

# plot the deviation from the point like approximation and jetset in the bottom panel
deviation_approx = sed_ec_out / sed_ec_ps - 1
deviation_jetset_out = sed_ec_out / sed_ec_out_jetset - 1

ax4.grid(False)
ax4.axhline(0, ls="-", color="darkgray")
ax4.axhline(0.2, ls="--", color="darkgray")
ax4.axhline(-0.2, ls="--", color="darkgray")
ax4.axhline(0.3, ls="-.", color="darkgray")
ax4.axhline(-0.3, ls="-.", color="darkgray")
ax4.axhline(0.5, ls=":", color="darkgray")
ax4.axhline(-0.5, ls=":", color="darkgray")
ax4.set_ylim([-1.1, 1.1])
ax4.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax4.semilogx(
    nu_ec, deviation_approx, ls="--", color="k", label="point-source approximation",
)
ax4.semilogx(nu_ec, deviation_jetset_out, ls="--", color="dodgerblue", label="jetset")
ax4.legend(loc="best", fontsize=10)
ax4.set_xlabel(sed_x_label)

Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_9.png")
fig.savefig(f"figures/figure_9.pdf")
