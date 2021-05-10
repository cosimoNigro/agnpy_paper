# import numpy, astropy and matplotlib for basic functionalities
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from pathlib import Path
from astropy.table import Table
import matplotlib.pyplot as plt
import pkg_resources

# import agnpy classes
from agnpy.emission_regions import Blob
from agnpy.spectra import BrokenPowerLaw
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label

# import sherpa classes
from sherpa.models import model
from sherpa import data
from sherpa.fit import Fit
from sherpa.stats import Chi2
from sherpa.optmethods import LevMar
from sherpa.estmethods import Confidence


class BrokenPowerLawSSC(model.RegriddableModel1D):
    """wrapper of agnpy's synchrotron and SSC classes. A log-parabola is assumed for the electron spectrum.
    The log10 of the normalisation and Lorentz factors (and magnetic field) are feeded to this model"""

    def __init__(self, name="ssc"):

        # EED parameters
        self.log10_k_e = model.Parameter(name, "log10_k_e", -2.0, min=-20.0, max=2.0)
        self.p1 = model.Parameter(name, "p1", 2.1, min=1.0, max=5.0)
        self.p2 = model.Parameter(name, "p2", 3.1, min=1.0, max=5.0)
        self.log10_gamma_b = model.Parameter(
            name, "log10_gamma_b", 3.0, min=1.0, max=6.0
        )
        self.log10_gamma_min = model.Parameter(
            name, "log10_gamma_min", 1.0, min=0.0, max=4.0
        )
        self.log10_gamma_max = model.Parameter(
            name, "log10_gamma_max", 5.0, min=3.0, max=8.0
        )
        # source general parameters
        self.z = model.Parameter(name, "z", 0.1, min=0.01, max=1)
        self.d_L = model.Parameter(name, "d_L", 1e27, min=1e25, max=1e33)
        # emission region parameters
        self.delta_D = model.Parameter(name, "delta_D", 10, min=0, max=40)
        self.log10_B = model.Parameter(name, "log10_B", -2, min=-4, max=2)
        self.log10_R_b = model.Parameter(name, "log10_R_b", 16, min=14, max=18)

        model.RegriddableModel1D.__init__(
            self,
            name,
            (
                self.log10_k_e,
                self.p1,
                self.p2,
                self.log10_gamma_b,
                self.log10_gamma_min,
                self.log10_gamma_max,
                self.z,
                self.d_L,
                self.delta_D,
                self.log10_B,
                self.log10_R_b,
            ),
        )

    def calc(self, pars, x):
        """evaluate the model calling the agnpy functions"""
        (
            log10_k_e,
            p1,
            p2,
            log10_gamma_b,
            log10_gamma_min,
            log10_gamma_max,
            z,
            d_L,
            delta_D,
            log10_B,
            log10_R_b,
        ) = pars
        # add units, scale quantities
        x *= u.Hz
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** (log10_B) * u.G
        R_b = 10 ** (log10_R_b) * u.cm
        d_L *= u.cm

        sed_synch = Synchrotron.evaluate_sed_flux(
            x,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            x,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
        )
        return sed_synch + sed_ssc


# read the 1D data
sed_path = pkg_resources.resource_filename("agnpy", "data/mwl_seds/Mrk421_2011.ecsv")
sed_table = Table.read(sed_path)
x = sed_table["nu"]
y = sed_table["nuFnu"]
y_err = sed_table["nuFnu_err"]
# remove the points with orders of magnitude smaller error, they are upper limits
UL = y_err < (y * 1e-3)
# add an arbitrary systematic error of 10% on the flux of all points
syst_err = 0.1 * y
# load the SED points in the sherpa data object
sed = data.Data1D("sed", x[~UL], y[~UL], staterror=y_err[~UL], syserror=syst_err[~UL])

# parameters from Table 4 and Figure 11 of Abdo 2011
R_b = 5.2 * 1e16 * u.cm
z = 0.0308
d_L = Distance(z=z).to("cm")
B = 3.8 * 1e-2 * u.G
# instance of the model wrapping angpy functionalities
# load and set all the blob parameters
model = BrokenPowerLawSSC()
# - AGN parameters
model.z = z
model.z.freeze()
model.d_L = d_L.cgs.value
model.d_L.freeze()
# - blob parameters
model.log10_R_b = np.log10(R_b.to_value("cm"))
model.log10_R_b.freeze()
model.delta_D = 20
model.delta_D.freeze()
model.log10_B = np.log10(B.to_value("G"))
model.log10_B.freeze()
# - EED
model.log10_k_e = -5.5
model.log10_gamma_b = np.log10(1e4)
model.p1 = 1.8
model.p2 = 2.9
model.log10_gamma_min = np.log10(500)
model.log10_gamma_min.freeze()
model.log10_gamma_max = np.log10(1e6)
model.log10_gamma_max.freeze()
print(model)

# fit using the Levenberg-Marquardt optimiser
fitter = Fit(sed, model, stat=Chi2(), method=LevMar())
# use confidence to estimate the errors
fitter.estmethod = Confidence()
fitter.estmethod.parallel = True
min_x = 1e11
max_x = 1e30
sed.notice(min_x, max_x)

# perform the first fit, we are only varying the spectral parameters
print("-- first iteration with only spectral parameters free")
results_1 = fitter.fit()
print("-- fit succesful?", results_1.succeeded)
print(results_1.format())

# perform the second fit, we are varying also the blob parameters
print("-- second iteration with spectral and blob parameters free")
model.log10_R_b.thaw()
model.delta_D.thaw()
model.log10_B.thaw()
results_2 = fitter.fit()
errors_2 = fitter.est_errors()
print("-- fit succesful?", results_2.succeeded)
print(results_2.format())
print("-- errors estimation:")
print(errors_2.format())
# plot the final model
nu = np.logspace(9, 30, 200)
plt.errorbar(sed.x, sed.y, yerr=sed.get_error(), ls="", marker=".", color="k")
plt.loglog(nu, model(nu), color="crimson")
plt.ylim([1e-14, 1e-9])
plt.show()


# plot the best fit model with the individual components
k_e = 10 ** model.log10_k_e.val * u.Unit("cm-3")
p1 = model.p1.val
p2 = model.p2.val
gamma_b = 10 ** model.log10_gamma_b.val
gamma_min = 10 ** model.log10_gamma_min.val
gamma_max = 10 ** model.log10_gamma_max.val
B = 10 ** model.log10_B.val * u.G
R_b = 10 ** model.log10_R_b.val * u.cm
delta_D = model.delta_D.val
parameters = {
    "p1": p1,
    "p2": p2,
    "gamma_b": gamma_b,
    "gamma_min": gamma_min,
    "gamma_max": gamma_max,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
blob = Blob(
    R_b, z, delta_D, delta_D, B, k_e, spectrum_dict, spectrum_norm_type="differential"
)
synch = Synchrotron(blob)
ssc = SynchrotronSelfCompton(blob)
# make a finer grid to compute the SED
nu = np.logspace(10, 30, 300) * u.Hz
synch_sed = synch.sed_flux(nu)
ssc_sed = ssc.sed_flux(nu)

load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.errorbar(
    sed.x,
    sed.y,
    yerr=sed.get_error(),
    marker=".",
    ls="",
    color="k",
    label="Mrk 421, Abdo et al. (2011)",
)
ax.loglog(
    nu, synch_sed + ssc_sed, ls="-", lw=2.1, color="crimson", label="agnpy, total"
)
ax.loglog(nu, synch_sed, ls="--", lw=1.3, color="goldenrod", label="agnpy, synchrotron")
ax.loglog(nu, ssc_sed, ls="--", lw=1.3, color="dodgerblue", label="agnpy, SSC")
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_ylim([1e-14, 1e-9])
ax.legend(loc="best")
plt.show()
fig.savefig("figures/Mrk421_fit.png")
fig.savefig("figures/Mrk421_fit.pdf")
