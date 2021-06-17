# general modules
import time
import logging
import warnings
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_e, c, G, M_sun
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt

# agnpy modules
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus
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

# constants
mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_size = 400
gamma_to_integrate = np.logspace(0, 7, gamma_size)


class AgnpyEC(SpectralModel):
    """Wrapper of agnpy's non synchrotron, SSC and EC classes. The flux model
    accounts for the Disk and DT's thermal SEDs. 
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters 
    whose range is expected to cover several orders of magnitudes (normalisation, 
    gammas, size and magnetic field of the blob). 
    """

    tag = "EC"
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
    delta_D = Parameter("delta_D", 10, min=1, max=40)
    mu_s = Parameter("mu_s", 0.9, min=0.0, max=1.0)
    log10_B = Parameter("log10_B", 0.0, min=-3.0, max=1.0)
    alpha_jet = Parameter("alpha_jet", 0.05, min=0.0, max=1.1)
    log10_r = Parameter("log10_r", 17.0, min=16.0, max=20.0)
    # disk parameters
    log10_L_disk = Parameter("log10_L_disk", 45.0, min=42.0, max=48.0)
    log10_M_BH = Parameter("log10_M_BH", 42, min=32, max=45)
    m_dot = Parameter("m_dot", 1e26, min=1e24, max=1e30)
    R_in = Parameter("R_in", 1e14, min=1e12, max=1e16)
    R_out = Parameter("R_out", 1e17, min=1e12, max=1e19)
    # DT parameters
    xi_dt = Parameter("xi_dt", 0.6, min=0.0, max=1.0)
    T_dt = Parameter("T_dt", 1.0e3, min=1.0e2, max=1.0e4)
    R_dt = Parameter("R_dt", 2.5e18, min=1.0e17, max=1.0e19)

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
        mu_s,
        log10_B,
        alpha_jet,
        log10_r,
        log10_L_disk,
        log10_M_BH,
        m_dot,
        R_in,
        R_out,
        xi_dt,
        T_dt,
        R_dt,
    ):
        # conversion
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
        M_BH = 10 ** log10_M_BH * u.Unit("g")
        m_dot *= u.Unit("g s-1")
        R_in *= u.cm
        R_out *= u.cm
        R_dt *= u.cm
        T_dt *= u.K
        r = 10 ** log10_r * u.cm
        R_b = r * alpha_jet
        eps_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        nu = energy.to("Hz", equivalencies=u.spectral())
        # non-thermal components
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
            ssa=True,
            gamma=gamma_to_integrate,
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
            ssa=True,
            gamma=gamma_to_integrate,
        )
        sed_ec_dt = ExternalCompton.evaluate_sed_flux_dt(
            nu,
            z,
            d_L,
            delta_D,
            mu_s,
            R_b,
            L_disk,
            xi_dt,
            eps_dt,
            R_dt,
            r,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
            gamma=gamma_to_integrate,
        )
        # thermal components
        sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
            nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
            nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
        )
        sed = sed_synch + sed_ssc + sed_ec_dt + sed_bb_disk + sed_bb_dt
        return (sed / energy ** 2).to("1 / (cm2 eV s)")


SPECTRAL_MODEL_REGISTRY.append(AgnpyEC)


logging.info("reading PKS1510-089 SED from agnpy datas")
sed_path = pkg_resources.resource_filename(
    "agnpy", "data/mwl_seds/PKS1510-089_2015.ecsv"
)
sed_table = Table.read(sed_path)
x = sed_table["E"].to("eV")
y = sed_table["nuFnu"].to("erg cm-2 s-1")
y_err = sed_table["nuFnu_err_lo"].to("erg cm-2 s-1")
# store in a Table readable by gammapy's FluxPoints
flux_points_table = Table()
flux_points_table["e_ref"] = x
flux_points_table["e2dnde"] = y
flux_points_table["e2dnde_err"] = y_err
flux_points_table.meta["SED_TYPE"] = "e2dnde"
flux_points = FluxPoints(flux_points_table)
flux_points = flux_points.to_sed_type("dnde")

# declare a model
agnpy_ec = AgnpyEC()
# global parameters of the blob and the DT
z = 0.361
d_L = Distance(z=z).to("cm")
# blob
Gamma = 20
alpha_jet = 0.047  # jet opening angle
delta_D = 25
Beta = np.sqrt(1 - 1 / np.power(Gamma, 2))  # jet relativistic speed
mu_s = (1 - 1 / (Gamma * delta_D)) / Beta  # viewing angle
B = 0.35 * u.G
# disk
L_disk = 6.7e45 * u.Unit("erg s-1")  # disk luminosity
M_BH = 5.71 * 1e7 * M_sun
eta = 1 / 12
m_dot = (L_disk / (eta * c ** 2)).to("g s-1")
R_g = ((G * M_BH) / c ** 2).to("cm")
R_in = 6 * R_g
R_out = 10000 * R_g
# DT
xi_dt = 0.6  # fraction of disk luminosity reprocessed by the DT
R_dt = 6.5 * 1e18 * u.cm  # radius of DT
T_dt = 1e3 * u.K
# location of the emission region
r = 7e17 * u.cm
# instance of the model wrapping angpy functionalities
# - AGN parameters
# -- distances
agnpy_ec.z.quanity = z
agnpy_ec.z.frozen = True
agnpy_ec.d_L.quantity = d_L.cgs.value
agnpy_ec.d_L.frozen = True
# -- SS disk
agnpy_ec.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
agnpy_ec.log10_L_disk.frozen = True
agnpy_ec.log10_M_BH.quantity = np.log10(M_BH.to_value("g"))
agnpy_ec.log10_M_BH.frozen = True
agnpy_ec.m_dot.quantity = m_dot.to_value("g s-1")
agnpy_ec.m_dot.frozen = True
agnpy_ec.R_in.quantity = R_in.to_value("cm")
agnpy_ec.R_in.frozen = True
agnpy_ec.R_out.quantity = R_out.to_value("cm")
agnpy_ec.R_out.frozen = True
# -- Dust Torus
agnpy_ec.xi_dt.quantity = xi_dt
agnpy_ec.xi_dt.frozen = True
agnpy_ec.T_dt.quantity = T_dt.to_value("K")
agnpy_ec.T_dt.frozen = True
agnpy_ec.R_dt.quantity = R_dt.to_value("cm")
agnpy_ec.R_dt.frozen = True
# - blob parameters
agnpy_ec.delta_D.quantity = delta_D
agnpy_ec.delta_D.frozen = True
agnpy_ec.mu_s.quantity = mu_s
agnpy_ec.mu_s.frozen = True
agnpy_ec.alpha_jet.quantity = alpha_jet
agnpy_ec.alpha_jet.frozen = True
agnpy_ec.log10_r.quantity = np.log10(r.to_value("cm"))
agnpy_ec.log10_r.frozen = True
agnpy_ec.log10_B.quantity = np.log10(B.to_value("G"))
agnpy_ec.log10_B.frozen = True
# - EED
agnpy_ec.log10_k_e.quantity = np.log10(0.05)
agnpy_ec.p1.quantity = 1.8
agnpy_ec.p2.quantity = 3.5
agnpy_ec.log10_gamma_b.quantity = np.log10(500)
agnpy_ec.log10_gamma_min.quantity = np.log10(1)
agnpy_ec.log10_gamma_min.frozen = True
agnpy_ec.log10_gamma_max.quantity = np.log10(3e4)
agnpy_ec.log10_gamma_max.frozen = True

# define model
model = SkyModel(name="PKS1510-089_EC", spectral_model=agnpy_ec)
dataset_ec = FluxPointsDataset(model, flux_points)
# do not use frequency point below 1e11 Hz, affected by non-blazar emission
E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
dataset_ec.mask_fit = dataset_ec.data.energy_ref > E_min_fit
logging.info(f"flux points dataset shape: {dataset_ec.data_shape()}")

# fit
logging.info("first fit iteration with only EED parameters thawed")
fitter = Fit([dataset_ec])
t_start_1 = time.perf_counter()
result_1 = fitter.run(optimize_opts={"print_level": 1})
t_stop_1 = time.perf_counter()
delta_t_1 = t_stop_1 - t_start_1
logging.info(f"time elapsed first fit: {delta_t_1:.2f} s")
print(result_1)
print(agnpy_ec.parameters.to_table())

logging.info("second fit iteration with EED and blob parameters thawed")
agnpy_ec.log10_r.frozen = False
agnpy_ec.log10_B.frozen = False
t_start_2 = time.perf_counter()
result_2 = fitter.run(optimize_opts={"print_level": 1})
t_stop_2 = time.perf_counter()
delta_t_2 = t_stop_2 - t_start_2
logging.info(f"time elapsed first fit: {delta_t_2:.2f} s")
print(result_2)
print(agnpy_ec.parameters.to_table())

logging.info("generating diagnostic plots for the fit")
fit_check_dir = "figures/figure_7_fit_check"
Path(fit_check_dir).mkdir(parents=True, exist_ok=True)
# best-fit model
flux_points.plot(energy_unit="eV", energy_power=2)
agnpy_ec.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
plt.ylim([10 ** (-13.5), 10 ** (-7.5)])
plt.savefig(f"{fit_check_dir}/best_fit.png")
plt.close()
# print and plot covariance
agnpy_ec.covariance.plot_correlation()
plt.savefig(f"{fit_check_dir}/correlation_matrix.png")
plt.close()
# chi2 profiles
total_stat = result_2.total_stat
for par in dataset_ec.models.parameters:
    if par.frozen is False:
        profile = fitter.stat_profile(parameter=par)
        plt.plot(profile[f"{par.name}_scan"], profile["stat_scan"] - total_stat)
        plt.xlabel(f"{par.unit}")
        plt.ylabel(r"$\chi^2$")
        plt.title(f"{par.name}: {par.value} +- {par.error}")
        plt.savefig(f"{fit_check_dir}/chi2_profile_parameter_{par.name}.png")
        plt.close()


logging.info("plot the final model with the individual components")
# plot the best fit model with the individual components
k_e = 10 ** agnpy_ec.log10_k_e.value * u.Unit("cm-3")
p1 = agnpy_ec.p1.value
p2 = agnpy_ec.p2.value
gamma_b = 10 ** agnpy_ec.log10_gamma_b.value
gamma_min = 10 ** agnpy_ec.log10_gamma_min.value
gamma_max = 10 ** agnpy_ec.log10_gamma_max.value
B = 10 ** agnpy_ec.log10_B.value * u.G
r = 10 ** agnpy_ec.log10_r.value * u.cm
delta_D = agnpy_ec.delta_D.value
R_b = r * alpha_jet
# blob definition
parameters = {
    "p1": p1,
    "p2": p2,
    "gamma_b": gamma_b,
    "gamma_min": gamma_min,
    "gamma_max": gamma_max,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
blob = Blob(
    R_b,
    z,
    delta_D,
    Gamma,
    B,
    k_e,
    spectrum_dict,
    spectrum_norm_type="differential",
    gamma_size=500,
)
# Disk and DT definition
L_disk = 10 ** agnpy_ec.log10_L_disk.value * u.Unit("erg s-1")
M_BH = 10 ** agnpy_ec.log10_M_BH.value * u.Unit("g")
m_dot = agnpy_ec.m_dot.value * u.Unit("g s-1")
eta = (L_disk / (m_dot * c ** 2)).to_value("")
R_in = agnpy_ec.R_in.value * u.cm
R_out = agnpy_ec.R_out.value * u.cm
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
dt = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)

# radiative processes
synch = Synchrotron(blob, ssa=True)
ssc = SynchrotronSelfCompton(blob, synch)
ec_dt = ExternalCompton(blob, dt, r)
# SEDs
nu = np.logspace(9, 27, 200) * u.Hz
synch_sed = synch.sed_flux(nu)
ssc_sed = ssc.sed_flux(nu)
ec_dt_sed = ec_dt.sed_flux(nu)
disk_bb_sed = disk.sed_flux(nu, z)
dt_bb_sed = dt.sed_flux(nu, z)
total_sed = synch_sed + ssc_sed + ec_dt_sed + disk_bb_sed + dt_bb_sed

load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.loglog(
    nu / (1 + z), total_sed, ls="-", lw=2.1, color="crimson", label="agnpy, total"
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
ax.loglog(
    nu / (1 + z), ec_dt_sed, ls="--", lw=1.3, color="seagreen", label="agnpy, EC on DT"
)
ax.loglog(
    nu / (1 + z),
    disk_bb_sed,
    ls="-.",
    lw=1.3,
    color="dimgray",
    label="agnpy, disk blackbody",
)
ax.loglog(
    nu / (1 + z),
    dt_bb_sed,
    ls=":",
    lw=1.3,
    color="dimgray",
    label="agnpy, DT blackbody",
)
ax.errorbar(
    flux_points_table["e_ref"].to("Hz", equivalencies=u.spectral()).data,
    flux_points_table["e2dnde"].to("erg cm-2 s-1").data,
    yerr=flux_points_table["e2dnde_err"].to("erg cm-2 s-1").data,
    marker=".",
    ls="",
    color="k",
    label="PKS 1510-089, Ahnen et al. (2017), period B",
)
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_ylim([10 ** (-13.5), 10 ** (-7.5)])
ax.legend(
    loc="upper center", fontsize=10, ncol=2,
)
plt.show()
fig.savefig("figures/figure_7.png")
fig.savefig("figures/figure_7.pdf")
