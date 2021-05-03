import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.targets import RingDustTorus
from agnpy.compton import ExternalCompton
from agnpy.constraints import SpectralConstraints
from agnpy.utils.plot import load_mpl_rc, plot_sed
import matplotlib.pyplot as plt

# define the emission region and the radiative processes
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
blob.set_gamma_size(500)
# target
dt = RingDustTorus(2e46 * u.Unit("erg s-1"), 0.2, 1000 * u.K)
# array of frequencies to compute the SEDs
nu = np.logspace(15, 28, 100) * u.Hz

# plot
load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
# loop over three different distances
for r, color in zip(
    [1e19 * u.cm, 1e20 * u.cm, 1e21 * u.cm], ["k", "crimson", "dodgerblue"]
):
    ec = ExternalCompton(blob, dt, r)
    sed = ec.sed_flux(nu)
    _pow = str(int(np.log10(r.to_value("cm"))))
    plot_sed(
        nu, sed, ax=ax, lw=2, label=r"$r = 10^{" + _pow + r"}\,{\rm cm}$", color=color
    )
ax.legend(loc="best")
ax.set_title("EC on dust torus")
fig.savefig("figures/ec_dt_example.png")
fig.savefig("figures/ec_dt_example.pdf")

# print the constraint on the spectral parameters
constraints = SpectralConstraints(blob)
# max and break Lorentz factor due to synchrotron radiation
gamma_max_synch = constraints.gamma_max_synch
gamma_break_synch = constraints.gamma_break_synch
# max and break Lorentz factor due to SSC energy losses
gamma_max_ssc = constraints.gamma_max_SSC
gamma_break_ssc = constraints.gamma_break_SSC
# max and break Lorentz factor deu to EC energy losses
gamma_max_ec_dt = constraints.gamma_max_EC_DT(dt, r=1e19 * u.cm)
gamma_break_ec_dt = constraints.gamma_break_EC_DT(dt, r=1e19 * u.cm)

print(f"gamma_max_synch = {gamma_max_synch:.2e}")
print(f"gamma_break_synch = {gamma_break_synch:.2e}")
print(f"gamma_max_ssc = {gamma_max_ssc:.2e}")
print(f"gamma_break_ssc = {gamma_break_ssc:.2e}")
print(f"gamma_max_ec_dt = {gamma_max_ec_dt:.2e}")
print(f"gamma_break_ec_dt = {gamma_break_ec_dt:.2e}")
