# script to compare the EC on DT against Finke 2016 and jetset
import numpy as np
import astropy.units as u
import pkg_resources
import copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, RingDustTorus
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import sed_x_label, sed_y_label
from pathlib import Path
from utils import reproduce_sed, time_sed_flux


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
blob.set_gamma_size(600)

# for the scattering on point source we will need a blob with a denser grid in
# Lorentz factor, let us create a copy and increase the size of the gamma grid
blob_ps = copy.copy(blob)
blob_ps.set_gamma_size(1000)

# DT parameters of Finke 2016
L_disk = 2 * 1e46 * u.Unit("erg s-1")
T_dt = 1e3 * u.K
xi_dt = 0.1
dt = RingDustTorus(L_disk, xi_dt, T_dt)

# point sources approximating the DT at very large distances
ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)

# EC definition
# - near the DT, to be compared with the reference
r_near = 1e18 * u.cm
ec_near = ExternalCompton(blob, dt, r=r_near)

# - far from the DT, to be compared with the point-source approximation
r_far = 1e22 * u.cm
ec_far = ExternalCompton(blob, dt, r=r_far)
ec_ps = ExternalCompton(blob_ps, ps_dt, r=r_far)

nu_ec = np.logspace(16, 29, 40) * u.Hz
sed_ec_near = time_sed_flux(ec_near, nu_ec)
sed_ec_far = time_sed_flux(ec_far, nu_ec)
sed_ec_ps = time_sed_flux(ec_ps, nu_ec)

# reproduce Figure 11 of Finke 2016 with agnpy
data_ec_dt_finke = pkg_resources.resource_filename(
    "agnpy", "data/reference_seds/finke_2016/figure_11/ec_dt_r_1e18.txt"
)

ec_nu_range = [nu_ec[0], nu_ec[-1]]
nu_ref, sed_ref, sed_ec_near_finke = reproduce_sed(
    data_ec_dt_finke, ec_near, ec_nu_range
)


# jetset
from jetset.jet_model import Jet

jet = Jet(
    name="ec_dt",
    electron_distribution="bkn",
    electron_distribution_log_values=False,
    beaming_expr="bulk_theta",
)
jet.add_EC_component(["EC_DT"])

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

# - DT
jet.set_par("L_Disk", val=L_disk.value)
jet.set_par("tau_DT", val=dt.xi_dt)
jet.set_par("R_DT", val=dt.R_dt.value)
jet.set_par("T_DT", val=dt.T_dt.value)

# - integration setup
jet.electron_distribution.update()
jet.set_gamma_grid_size(10000)
jet._blob.IC_adaptive_e_binning = True
jet._blob.theta_n_int = 500
jet.set_nu_grid(nu_ec[0].value, nu_ec[-1].value, len(nu_ec))

# - SED near the DT
jet.set_par("R_H", val=r_near.to_value("cm"))
jet.set_external_field_transf("disk")
# fixes by Andrea to reproduce Finke's approach
jet._blob.R_H_scale_factor = 50
jet._blob.R_ext_factor = 0.5
theta_lim = np.rad2deg(np.arctan(jet.get_beaming() / jet.parameters.BulkFactor.val))
jet._blob.EC_theta_lim = theta_lim

jet.eval()

sed_ec_near_jetset = jet.spectral_components.EC_DT.SED.nuFnu

# - SED far from the DT
jet.set_par("R_H", val=r_far.to_value("cm"))
jet.set_external_field_transf("disk")
# fixes by Andrea to reproduce Finke's approach
jet._blob.R_H_scale_factor = 50
jet._blob.R_ext_factor = 0.5
theta_lim = np.rad2deg(np.arctan(jet.get_beaming() / jet.parameters.BulkFactor.val))
jet._blob.EC_theta_lim = theta_lim

jet.eval()

sed_ec_far_jetset = jet.spectral_components.EC_DT.SED.nuFnu


# make figure 10
# gridspec plot setting
fig = plt.figure(figsize=(12, 6), tight_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[2, 1], figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 0], sharex=ax1)
ax4 = fig.add_subplot(spec[1, 1], sharex=ax2, sharey=ax3)

# SED near the DT
ax1.loglog(nu_ref, sed_ec_near_finke, ls="-", lw=2.1, color="crimson", label="agnpy")
ax1.loglog(
    nu_ref, sed_ref, ls="--", color="k", label="Fig. 11, Finke (2016)",
)
ax1.loglog(nu_ec, sed_ec_near_jetset, ls="--", color="dodgerblue", label="jetset")
ax1.set_ylabel(sed_y_label)
ax1.legend(loc="best", fontsize=12)
ax1.set_title("EC on ring DT, " + r"$r=10^{18}\,{\rm cm} < R_{\rm DT}$", fontsize=15)

# SED far from the DT
ax2.loglog(
    nu_ec, sed_ec_far, ls="-", lw=2.1, color="crimson", label="agnpy, full calculation",
)
ax2.loglog(
    nu_ec, sed_ec_ps, ls="--", color="k", label="agnpy, point-source approximation",
)
ax2.loglog(nu_ec, sed_ec_far_jetset, ls="--", color="dodgerblue", label="jetset")
ax2.legend(loc="best", fontsize=12)
ax2.set_title("EC on ring DT, " + r"$r=10^{22}\,{\rm cm} \gg R_{\rm DT}$", fontsize=15)

# plot the deviation from the references in the bottom panel
deviation_ref = sed_ec_near_finke / sed_ref - 1
deviation_jetset_near = sed_ec_near / sed_ec_near_jetset - 1

ax3.grid(False)
ax3.axhline(0, ls="-", color="darkgray")
ax3.axhline(0.2, ls="--", color="darkgray")
ax3.axhline(-0.2, ls="--", color="darkgray")
ax3.axhline(0.5, ls=":", color="darkgray")
ax3.axhline(-0.5, ls=":", color="darkgray")
ax3.set_ylim([-1.1, 1.1])
ax3.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax3.semilogx(
    nu_ref, deviation_ref, ls="--", color="k", label="Fig. 10, Finke (2016)",
)
ax3.semilogx(
    nu_ec, deviation_jetset_near, ls="--", color="dodgerblue", label="jetset",
)
ax3.legend(loc="best", fontsize=11)
ax3.set_xlabel(sed_x_label)
ax3.set_ylabel(r"$\frac{\nu F_{\nu, \rm agnpy}}{\nu F_{\nu, \rm ref}} - 1$")

# plot the deviation from the point like approximation and jetset in the bottom panel
deviation_approx = sed_ec_far / sed_ec_ps - 1
deviation_jetset_out = sed_ec_far / sed_ec_far_jetset - 1

ax4.grid(False)
ax4.axhline(0, ls="-", color="darkgray")
ax4.axhline(0.2, ls="--", color="darkgray")
ax4.axhline(-0.2, ls="--", color="darkgray")
ax4.axhline(0.5, ls=":", color="darkgray")
ax4.axhline(-0.5, ls=":", color="darkgray")
ax4.set_ylim([-1.1, 1.1])
ax4.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax4.semilogx(
    nu_ec, deviation_approx, ls="--", color="k", label="point-source approximation",
)
ax4.semilogx(
    nu_ec, deviation_jetset_out, ls="--", color="dodgerblue", label="jetset",
)
ax4.legend(loc="best", fontsize=11)
ax4.set_xlabel(r"$\nu\,/\,{\rm Hz}$")

Path("figures").mkdir(exist_ok=True)
fig.savefig(f"figures/figure_10.png")
fig.savefig(f"figures/figure_10.pdf")
