import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.utils.plot import plot_sed, load_mpl_rc
import matplotlib.pyplot as plt

# define the emission region and the radiative process
blob = Blob()
synch = Synchrotron(blob)
# compute the SED over an array of frequencies
nu = np.logspace(8, 23) * u.Hz
sed = synch.sed_flux(nu)
# plot it
load_mpl_rc()
fig, ax = plt.subplots()
plot_sed(nu, sed, ax=ax, label="Synchrotron")
plt.show()
fig.savefig("figures/synchro.pdf")
