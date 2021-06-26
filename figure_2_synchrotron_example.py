import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.utils.plot import plot_sed, load_mpl_rc
import matplotlib.pyplot as plt
from pathlib import Path
from utils import time_function_call

# define the emission region and the radiative process
blob = Blob()
synch = Synchrotron(blob)
# compute the SED over an array of frequencies, time it
nu = np.logspace(8, 23, 100) * u.Hz
sed = time_function_call(synch.sed_flux, nu)

# plot
load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
plot_sed(nu, sed, ax=ax, color="k", lw=2, label="synchrotron")
ax.legend(loc="best")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/figure_2.png")
fig.savefig("figures/figure_2.pdf")
