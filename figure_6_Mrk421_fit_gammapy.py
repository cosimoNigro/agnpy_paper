# general modules
import time
import logging
import warnings
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.constants import c
from astropy.coordinates import Distance
import matplotlib.pyplot as plt

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


logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")


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
sed_table = Table.read(sed_path)
x = sed_table["nu"].to("eV", equivalencies=u.spectral())
y = sed_table["nuFnu"].to("erg cm-2 s-1")
y_err_stat = sed_table["nuFnu_err"].to("erg cm-2 s-1")
# array of systematic errors, will just be summed in quadrature to the statistical error
# we assume
# - 15% on gamma-ray instruments
# - 10% on lower waveband instruments
y_err_syst = np.zeros(len(x))
gamma = x > 0.1 * u.GeV
y_err_syst[gamma] = 0.15
y_err_syst[~gamma] = 0.10
y_err_syst = y * y_err_syst
# remove the points with orders of magnitude smaller error, they are upper limits
UL = y_err_stat < (y * 1e-3)
x = x[~UL]
y = y[~UL]
y_err_stat = y_err_stat[~UL]
y_err_syst = y_err_syst[~UL]
# store in a Table readable by gammapy's FluxPoints
flux_points_table = Table()
flux_points_table["e_ref"] = x
flux_points_table["e2dnde"] = y
flux_points_table["e2dnde_err"] = np.sqrt(y_err_stat ** 2 + y_err_syst ** 2)
flux_points_table.meta["SED_TYPE"] = "e2dnde"
flux_points = FluxPoints(flux_points_table)
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
agnpy_ssc.delta_D.frozen = True
agnpy_ssc.log10_B.quantity = -1.3
agnpy_ssc.log10_B.frozen = True
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
# plot initial model
dataset_ssc.plot_spectrum(energy_power=2, energy_unit="eV")
plt.show()


# directory to store the checks performed on the fit
fit_check_dir = "figures/figure_6_checks_gammapy_fit"
Path(fit_check_dir).mkdir(parents=True, exist_ok=True)
# define the fitter
fitter = Fit([dataset_ssc])
logging.info("first fit iteration with only EED parameters thawed")
t_start_1 = time.perf_counter()
result_1 = fitter.run(optimize_opts={"print_level": 1})
t_stop_1 = time.perf_counter()
delta_t_1 = t_stop_1 - t_start_1
logging.info(f"time elapsed first fit: {delta_t_1:.2f} s")
print(result_1)
print(agnpy_ssc.parameters.to_table())
# plot best-fit model and covariance
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ssc.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
plt.savefig(f"{fit_check_dir}/best_fit_first_iteration.png")
plt.close()
agnpy_ssc.covariance.plot_correlation()
plt.savefig(f"{fit_check_dir}/correlation_matrix_first_iteration.png")
plt.close()

import IPython

IPython.embed()

logging.info("second fit iteration with EED and blob parameters thawed")
agnpy_ssc.delta_D.frozen = False
agnpy_ssc.log10_B.frozen = False
t_start_2 = time.perf_counter()
result_2 = fitter.run(optimize_opts={"print_level": 1})
t_stop_2 = time.perf_counter()
delta_t_2 = t_stop_2 - t_start_2
logging.info(f"time elapsed second fit: {delta_t_2:.2f} s")
print(result_2)
print(agnpy_ssc.parameters.to_table())
# plot best-fit model and covariance
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ssc.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
plt.savefig(f"{fit_check_dir}/best_fit_second_iteration.png")
plt.close()
agnpy_ssc.covariance.plot_correlation()
plt.savefig(f"{fit_check_dir}/correlation_matrix_second_iteration.png")
plt.close()

logging.info("computing statistics profiles")
# chi2 profiles
total_stat = result_2.total_stat
logging.info(f"computing statistics profile")
for par in agnpy_ssc.parameters:
    if par.frozen is False:
        logging.info(f"computing statistics profile for {par.name}")
        t_start_profile = time.perf_counter()
        profile = fitter.stat_profile(parameter=par, reoptimize=True)
        t_stop_profile = time.perf_counter()
        delta_t_profile = t_stop_profile - t_start_profile
        logging.info(f"time elapsed profile computation: {delta_t_profile:.2f} s")
        # plot profile
        plt.plot(profile[f"{par.name}_scan"], profile["stat_scan"] - total_stat)
        plt.xlabel(f"{par.unit}")
        plt.ylabel(r"$\Delta\chi^2$")
        plt.axhline(1, ls="--", color="orange")
        plt.title(f"{par.name}: {par.value:.3f} +- {par.error:.3f}")
        plt.savefig(f"{fit_check_dir}/chi2_profile_parameter_{par.name}.png")
        plt.close()

logging.info(f"computing confidence intervals")
for par in agnpy_ssc.parameters:
    if par.frozen is False:
        logging.info(f"computing confidence interval for {par.name}")
        t_start_confidence = time.perf_counter()
        confidence = fitter.confidence(parameter=par, reoptimize=True)
        t_stop_confidence = time.perf_counter()
        delta_t_confidence = t_stop_confidence - t_start_confidence
        logging.info(f"time elapsed confidence computation: {delta_t_confidence:.2f} s")
        print(confidence)

logging.info("plot the final model with the individual components")
k_e = 10 ** agnpy_ssc.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ssc.p1.value
p2 = agnpy_ssc.p2.value
gamma_b = 10 ** agnpy_ssc.log10_gamma_b.value
gamma_min = 10 ** agnpy_ssc.log10_gamma_min.value
gamma_max = 10 ** agnpy_ssc.log10_gamma_max.value
B = 10 ** agnpy_ssc.log10_B.value * u.G
R_b = (
    c
    * agnpy_ssc.t_var.quantity
    * agnpy_ssc.delta_D.quantity
    / (1 + agnpy_ssc.z.quantity)
).to("cm")
delta_D = agnpy_ssc.delta_D.value
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
print("jet power in particles", blob.P_jet_e)
print("jet power in B", blob.P_jet_B)

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
ax.errorbar(
    flux_points_table["e_ref"].to("Hz", equivalencies=u.spectral()).data,
    flux_points_table["e2dnde"].to("erg cm-2 s-1").data,
    yerr=flux_points_table["e2dnde_err"].to("erg cm-2 s-1").data,
    marker=".",
    ls="",
    color="k",
    label="Mrk 421, Abdo et al. (2011)",
)
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_xlim([1e-14, 1e-9])
ax.set_ylim([1e-14, 1e-9])
ax.legend(loc="best")
fig.savefig("figures/figure_6_gammapy_fit.png")
fig.savefig("figures/figure_6_gammapy_fit.pdf")
