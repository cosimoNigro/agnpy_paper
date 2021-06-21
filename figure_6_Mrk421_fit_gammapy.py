# general modules
import time
import logging
import warnings
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import Table
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
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=2)
    p1 = Parameter("p1", 2.1, min=1.0, max=5.0)
    p2 = Parameter("p2", 3.1, min=1.0, max=5.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=6)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=3, max=8)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=40)
    log10_B = Parameter("log10_B", -1, min=-4, max=2)
    log10_R_b = Parameter("log10_R_b", 16, min=14, max=18)

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
        log10_R_b,
    ):
        # conversion
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = 10 ** log10_R_b * u.cm

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
y_err = sed_table["nuFnu_err"].to("erg cm-2 s-1")
# remove the points with orders of magnitude smaller error, they are upper limits
UL = y_err < (y * 1e-3)
# store in a Table readable by gammapy's FluxPoints
flux_points_table = Table()
flux_points_table["e_ref"] = x[~UL]
flux_points_table["e2dnde"] = y[~UL]
flux_points_table["e2dnde_err"] = y_err[~UL]
flux_points_table.meta["SED_TYPE"] = "e2dnde"
flux_points = FluxPoints(flux_points_table)
flux_points = flux_points.to_sed_type("dnde")

# declare a model
agnpy_ssc = AgnpySSC()
# initialise parameters
# parameters from Table 4 and Figure 11 of Abdo (2011)
R_b = 5.2 * 1e16 * u.cm
z = 0.0308
d_L = Distance(z=z).to("cm")
B = 3.8 * 1e-2 * u.G
# - AGN parameters
agnpy_ssc.z.quantity = z
agnpy_ssc.z.frozen = True
agnpy_ssc.d_L.quantity = d_L
agnpy_ssc.d_L.frozen = True
# - blob parameters
agnpy_ssc.log10_R_b.quantity = np.log10(R_b.to_value("cm"))
agnpy_ssc.log10_R_b.frozen = True
agnpy_ssc.delta_D.quantity = 20
agnpy_ssc.delta_D.frozen = True
agnpy_ssc.log10_B.quantity = np.log10(B.to_value("G"))
agnpy_ssc.log10_B.frozen = True
# - EED
agnpy_ssc.log10_k_e.quantity = -5.5
agnpy_ssc.log10_gamma_b.quantity = np.log10(1e4)
agnpy_ssc.p1.quantity = 1.8
agnpy_ssc.p2.quantity = 2.9
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
logging.info(f"flux points dataset shape: {dataset_ssc.data_shape()}")

# fit
logging.info("first fit iteration with only EED parameters thawed")
fitter = Fit([dataset_ssc])
t_start_1 = time.perf_counter()
result_1 = fitter.run(optimize_opts={"print_level": 1})
t_stop_1 = time.perf_counter()
delta_t_1 = t_stop_1 - t_start_1
logging.info(f"time elapsed first fit: {delta_t_1:.2f} s")
print(result_1)
print(agnpy_ssc.parameters.to_table())

logging.info("second fit iteration with EED and blob parameters thawed")
# agnpy_ssc.log10_R_b.frozen = False
agnpy_ssc.delta_D.frozen = False
agnpy_ssc.log10_B.frozen = False
t_start_2 = time.perf_counter()
result_2 = fitter.run(optimize_opts={"print_level": 1})
t_stop_2 = time.perf_counter()
delta_t_2 = t_stop_2 - t_start_2
logging.info(f"time elapsed second fit: {delta_t_2:.2f} s")
print(result_2)
print(agnpy_ssc.parameters.to_table())

logging.info("computing covariance matrix and statistics profiles")
fit_check_dir = "figures/figure_6_checks_gammapy_fit"
Path(fit_check_dir).mkdir(parents=True, exist_ok=True)
# best-fit model
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ssc.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
plt.savefig(f"{fit_check_dir}/best_fit.png")
plt.close()
# print and plot covariance
agnpy_ssc.covariance.plot_correlation()
plt.savefig(f"{fit_check_dir}/correlation_matrix.png")
plt.close()
# chi2 profiles
total_stat = result_2.total_stat
for reoptimize in (False, True):
    logging.info(f"computing statistics profile with reoptimization {reoptimize}")
    for par in dataset_ssc.models.parameters:
        if par.frozen is False:
            logging.info(f"computing statistics profile for {par.name}")
            t_start_profile = time.perf_counter()
            profile = fitter.stat_profile(parameter=par, reoptimize=reoptimize)
            t_stop_profile = time.perf_counter()
            delta_t_profile = t_stop_profile - t_start_profile
            logging.info(f"time elapsed profile computation: {delta_t_profile:.2f} s")
            plt.plot(profile[f"{par.name}_scan"], profile["stat_scan"] - total_stat)
            plt.xlabel(f"{par.unit}")
            plt.ylabel(r"$\chi^2$")
            plt.title(f"{par.name}: {par.value:.3f} +- {par.error:.3f}")
            reoptimized = str(reoptimize).lower()
            plt.savefig(
                f"{fit_check_dir}/chi2_profile_parameter_{par.name}_reoptimize_{reoptimized}.png"
            )
            plt.close()


logging.info("plot the final model with the individual components")
k_e = 10 ** agnpy_ssc.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ssc.p1.value
p2 = agnpy_ssc.p2.value
gamma_b = 10 ** agnpy_ssc.log10_gamma_b.value
gamma_min = 10 ** agnpy_ssc.log10_gamma_min.value
gamma_max = 10 ** agnpy_ssc.log10_gamma_max.value
B = 10 ** agnpy_ssc.log10_B.value * u.G
R_b = 10 ** agnpy_ssc.log10_R_b.value * u.cm
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
