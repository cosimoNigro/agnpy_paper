from jetset.jet_model import Jet
import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import sed_x_label, sed_y_label
import matplotlib.pyplot as plt

# agnpy
# - blob
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
# - disk
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6
R_out = 200
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)

# jet(set) with broken power-law electron distribution
jet = Jet(
    name="",
    electron_distribution="bkn",
    electron_distribution_log_values=False,
    beaming_expr="bulk_theta",
)
jet.add_EC_component(["EC_Disk"], disk_type="MultiBB")
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
# - disk
jet.set_par("L_Disk", val=L_disk.value)
jet.set_par("M_BH", val=(M_BH / M_sun).to_value(""))
jet.set_par("accr_eff", val=eta)
jet.set_par("R_inner_Sw", val=R_in)
jet.set_par("R_ext_Sw", val=R_out)
jet.electron_distribution.update()

# compare the two electron distributions
gamma_jetset = jet.electron_distribution.gamma_e
n_e_jetset = jet.electron_distribution.n_gamma_e
plt.loglog(blob.gamma, blob.n_e(blob.gamma), color="crimson", label="agnpy")
plt.loglog(gamma_jetset, n_e_jetset, ls="--", color="k", label="jetset")
plt.legend()
plt.xlabel(r"$\gamma'$")
plt.ylabel(r"$n'_{\rm e}$")
plt.show()

# set the same grid in frequency
nu = np.logspace(14, 30, 100) * u.Hz
jet.set_nu_grid(1e14, 1e30, 100)

# compare for different distances
fig, ax = plt.subplots(
    2, 4, figsize=(12, 8), sharex=True, sharey=True, tight_layout=True
)
for i, transformation in enumerate(["disk", "blob"]):
    for j, _r in enumerate([0.1, 1.1, 10, 100]):
        r = _r * disk.R_out
        # - agnpy EC
        ec = ExternalCompton(blob, disk, r=r)
        ec_sed_agnpy = ec.sed_flux(nu)
        # - jetset EC
        jet.set_par("R_H", val=r.to_value("cm"))
        jet.set_external_field_transf(transformation)
        jet.show_model()
        # evaluate and fetch the EC on disk component
        jet.eval()
        # fetch nu and nuFnu from jetset
        nu_jetset = jet.spectral_components.EC_Disk.SED.nu
        ec_sed_jetset = jet.spectral_components.EC_Disk.SED.nuFnu
        # eliminate extremly low values
        null_values = ec_sed_jetset.value < 1e-50
        ax[i][j].loglog(
            nu_jetset[~null_values],
            ec_sed_jetset[~null_values],
            ls="--",
            color="k",
            label="jetset",
        )
        ax[i][j].loglog(nu, ec_sed_agnpy, color="crimson", label="agnpy")
        ax[i][j].legend()
        ax[i][j].set_ylim([1e-23, 1e-8])
        text = (
            f"frame = {transformation}\n"
            + r"$r = $"
            + f"{_r}"
            + r"$\times R_{\rm out}$"
        )
        ax[i][j].text(1e20, 8e-23, text, bbox=dict(boxstyle="round", fc="w", alpha=0.5))
        ax[i][j].grid(ls=":")
# set labels
ax[1][0].set_xlabel(sed_x_label)
ax[1][1].set_xlabel(sed_x_label)
ax[1][2].set_xlabel(sed_x_label)
ax[1][3].set_xlabel(sed_x_label)
ax[0][0].set_ylabel(sed_y_label)
ax[1][0].set_ylabel(sed_y_label)
fig.suptitle("EC on Disk")
fig.savefig(f"jetset_ec_disk_comparisons.png")
fig.savefig(f"jetset_ec_disk_comparisons.pdf")