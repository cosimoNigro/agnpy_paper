# general modules
import logging
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from utils import time_function_call

# agnpy modules
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label

# gammapy modules
from gammapy.modeling.models import (
    SpectralModel,
    Parameter,
    SPECTRAL_MODEL_REGISTRY,
    SkyModel,
)
from gammapy.estimators import FluxPoints
from gammapy.datasets import FluxPointsDataset
from gammapy.modeling import Fit


class AgnpySSC(SpectralModel):
    """Wrapper of agnpy's synchrotron and SSC classes. 
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters 
    whose range is expected to cover several orders of magnitudes (normalisation, 
    gammas, size and magnetic field of the blob). 
    """

    tag = "SSC"
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=10)
    p1 = Parameter("p1", 2.1, min=-2.0, max=5.0)
    p2 = Parameter("p2", 3.1, min=-2.0, max=5.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=6)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=4, max=8)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=40)
    log10_B = Parameter("log10_B", -1, min=-4, max=2)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)

    @staticmethod
    def evaluate(
        energy,
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
        t_var,
    ):
        # conversions
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")

        nu = energy.to("Hz", equivalencies=u.spectral())
        sed_synch = Synchrotron.evaluate_sed_flux(
            nu,
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
            nu,
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
        sed = sed_synch + sed_ssc
        return (sed / energy ** 2).to("1 / (cm2 eV s)")


SPECTRAL_MODEL_REGISTRY.append(AgnpySSC)


logging.info("reading Mrk421 SED from agnpy datas")
sed_path = pkg_resources.resource_filename("agnpy", "data/mwl_seds/Mrk421_2011.ecsv")
flux_points = FluxPoints.read(sed_path)
# array of systematic errors, will just be summed in quadrature to the statistical error
# we assume
# - 30% on VHE gamma-ray instruments
# - 10% on HE gamma-ray instruments
# - 10% on X-ray instruments
# - 5% on lower-energy instruments
x = flux_points.table["e_ref"]
y = flux_points.table["e2dnde"]
y_err_stat = flux_points.table["e2dnde_err"]
y_err_syst = np.zeros(len(x))
# define energy ranges
e_vhe = 100 * u.GeV
e_he = 0.1 * u.GeV
e_x_ray_max = 300 * u.keV
e_x_ray_min = 0.3 * u.keV
vhe_gamma = x >= e_vhe
he_gamma = (x >= e_he) * (x < e_vhe)
x_ray = (x >= e_x_ray_min) * (x < e_x_ray_max)
uv_to_radio = x < e_x_ray_min
# declare systematics
y_err_syst[vhe_gamma] = 0.30
y_err_syst[he_gamma] = 0.10
y_err_syst[x_ray] = 0.10
y_err_syst[uv_to_radio] = 0.05
y_err_syst = y * y_err_syst
# sum in quadrature the errors
flux_points.table["e2dnde_err"] = np.sqrt(y_err_stat ** 2 + y_err_syst ** 2)
flux_points = flux_points.to_sed_type("dnde")

# declare a model
agnpy_ssc = AgnpySSC()
# initialise parameters
z = 0.0308
d_L = Distance(z=z).to("cm")
# - AGN parameters
agnpy_ssc.z.quantity = z
agnpy_ssc.z.frozen = True
agnpy_ssc.d_L.quantity = d_L
agnpy_ssc.d_L.frozen = True
# - blob parameters
agnpy_ssc.delta_D.quantity = 18
agnpy_ssc.log10_B.quantity = -1.3
agnpy_ssc.t_var.quantity = 1 * u.d
agnpy_ssc.t_var.frozen = True
# - EED
agnpy_ssc.log10_k_e.quantity = -7.9
agnpy_ssc.p1.quantity = 2.02
agnpy_ssc.p2.quantity = 3.43
agnpy_ssc.log10_gamma_b.quantity = 5
agnpy_ssc.log10_gamma_min.quantity = np.log10(500)
agnpy_ssc.log10_gamma_min.frozen = True
agnpy_ssc.log10_gamma_max.quantity = np.log10(1e6)
agnpy_ssc.log10_gamma_max.frozen = True

# define model
model = SkyModel(name="Mrk421_SSC", spectral_model=agnpy_ssc)
dataset_ssc = FluxPointsDataset(model, flux_points)
# do not use frequency point below 1e11 Hz, affected by non-blazar emission
E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
dataset_ssc.mask_fit = dataset_ssc.data.energy_ref > E_min_fit

logging.info("performing the fit")
# directory to store the checks performed on the fit
fit_check_dir = "figures/figure_6_checks_gammapy_fit"
Path(fit_check_dir).mkdir(parents=True, exist_ok=True)
# define the fitter
fitter = Fit([dataset_ssc])
results = time_function_call(fitter.run, optimize_opts={"print_level": 1})
print(results)
print(agnpy_ssc.parameters.to_table())
# plot best-fit model and covariance
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ssc.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
plt.savefig(f"{fit_check_dir}/best_fit.png")
plt.close()
agnpy_ssc.covariance.plot_correlation()
plt.savefig(f"{fit_check_dir}/correlation_matrix.png")
plt.close()

logging.info("plot the final model with the individual components")
k_e = 10 ** agnpy_ssc.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ssc.p1.value
p2 = agnpy_ssc.p2.value
gamma_b = 10 ** agnpy_ssc.log10_gamma_b.value
gamma_min = 10 ** agnpy_ssc.log10_gamma_min.value
gamma_max = 10 ** agnpy_ssc.log10_gamma_max.value
B = 10 ** agnpy_ssc.log10_B.value * u.G
delta_D = agnpy_ssc.delta_D.value
R_b = (
    c
    * agnpy_ssc.t_var.quantity
    * agnpy_ssc.delta_D.quantity
    / (1 + agnpy_ssc.z.quantity)
).to("cm")
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
print(blob)
print(f"jet power in particles: {blob.P_jet_e:.2e}")
print(f"jet power in B: {blob.P_jet_B:.2e}")

# compute the obtained emission region
synch = Synchrotron(blob)
ssc = SynchrotronSelfCompton(blob)
# make a finer grid to compute the SED
nu = np.logspace(10, 30, 300) * u.Hz
synch_sed = synch.sed_flux(nu)
ssc_sed = ssc.sed_flux(nu)

load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.loglog(
    nu / (1 + z),
    synch_sed + ssc_sed,
    ls="-",
    lw=2.1,
    color="crimson",
    label="agnpy, total",
)
ax.loglog(
    nu / (1 + z),
    synch_sed,
    ls="--",
    lw=1.3,
    color="goldenrod",
    label="agnpy, synchrotron",
)
ax.loglog(
    nu / (1 + z), ssc_sed, ls="--", lw=1.3, color="dodgerblue", label="agnpy, SSC"
)
# systematics error in gray
ax.errorbar(
    x.to("Hz", equivalencies=u.spectral()).value,
    y,
    yerr=y_err_syst,
    marker=",",
    ls="",
    color="gray",
    label="",
)
# statistics error in black
ax.errorbar(
    x.to("Hz", equivalencies=u.spectral()).value,
    y,
    yerr=y_err_stat,
    marker=".",
    ls="",
    color="k",
    label="Mrk 421, Abdo et al. (2011)",
)
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_xlim([1e9, 1e29])
ax.set_ylim([1e-14, 1e-9])
ax.legend(loc="best")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/figure_6_gammapy_fit.png")
fig.savefig("figures/figure_6_gammapy_fit.pdf")
