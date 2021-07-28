from jetset.jet_model import Jet
import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.emission_regions import Blob
from agnpy.targets import CMB
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
# - CMB
cmb = CMB(z=blob.z)


# jet(set) with broken power-law electron distribution
jet = Jet(name="", electron_distribution="bkn", electron_distribution_log_values=False)
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
jet.set_par("beam_obj", val=blob.Gamma)
jet.set_par("z_cosm", val=blob.z)
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

# - agnpy EC
ec = ExternalCompton(blob, cmb)
ec_sed_agnpy = ec.sed_flux(nu)

# - jetset EC
jet.set_external_field_transf("disk")
jet.show_model()
# evaluate and fetch the EC on disk component
jet.eval()

nu_jetset = jet.spectral_components.EC_CMB.SED.nu
ec_sed_jetset = jet.spectral_components.EC_CMB.SED.nuFnu

fig, ax = plt.subplots()
ax.loglog(nu_jetset, ec_sed_jetset, ls="--", color="k", label="jetset")
ax.loglog(nu, ec_sed_agnpy, color="crimson", label="agnpy")
ax.legend()
ax.set_title(f"EC on CMB")
ax.set_ylim([1e-28, 1e-20])
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
plt.show()
fig.savefig(f"jetset_ec_cmb_comparison.png")
