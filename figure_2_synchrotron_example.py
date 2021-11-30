import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.utils.plot import plot_sed, load_mpl_rc
import matplotlib.pyplot as plt
from pathlib import Path
from utils import time_sed_flux

# define the emission region
# set the spectrum normalisation (total energy in electrons in this case)
spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function parametrisation through a dictionary
spectrum_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7},
}
# set the remaining quantities defining the blob
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# define the radiative process
synch = Synchrotron(blob)

# compute the SED over an array of frequencies, time it
nu = np.logspace(8, 23, 100) * u.Hz
sed = time_sed_flux(synch, nu)


# plot
load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()

plot_sed(nu, sed, ax=ax, color="k", lw=2, label="synchrotron")
ax.legend(loc="best")

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/figure_2.png")
fig.savefig("figures/figure_2.pdf")
