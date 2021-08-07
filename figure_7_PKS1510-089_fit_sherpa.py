# general modules
import logging
import pkg_resources
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_e, c, G, M_sun
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from utils import time_function_call

# agnpy modules
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label

# import sherpa classes
from sherpa.models import model
from sherpa import data
from sherpa.fit import Fit
from sherpa.stats import Chi2
from sherpa.optmethods import LevMar
from sherpa.estmethods import Confidence
from sherpa.plot import IntervalProjection


# constants
mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_size = 400
gamma_to_integrate = np.logspace(0, 7, gamma_size)


class AgnpyEC(model.RegriddableModel1D):
    """Wrapper of agnpy's non synchrotron, SSC and EC classes. The flux model
    accounts for the Disk and DT's thermal SEDs. 
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters 
    whose range is expected to cover several orders of magnitudes (normalisation, 
    gammas, size and magnetic field of the blob). 
    """

    def __init__(self, name="ec"):

        # EED parameters
        self.log10_k_e = model.Parameter(name, "log10_k_e", -2.0, min=-20.0, max=10.0)
        self.p1 = model.Parameter(name, "p1", 2.1, min=-2.0, max=5.0)
        self.p2 = model.Parameter(name, "p2", 3.1, min=-2.0, max=5.0)
        self.log10_gamma_b = model.Parameter(name, "log10_gamma_b", 3, min=1, max=6)
        self.log10_gamma_min = model.Parameter(name, "log10_gamma_min", 1, min=0, max=4)
        self.log10_gamma_max = model.Parameter(name, "log10_gamma_max", 5, min=4, max=8)
        # source general parameters
        self.z = model.Parameter(name, "z", 0.1, min=0.01, max=1)
        self.d_L = model.Parameter(name, "d_L", 1e27, min=1e25, max=1e33, units="cm")
        # emission region parameters
        self.delta_D = model.Parameter(name, "delta_D", 10, min=0, max=40)
        self.log10_B = model.Parameter(name, "log10_B", -2, min=-4, max=2)
        self.t_var = model.Parameter(
            name, "t_var", 600, min=10, max=np.pi * 1e7, units="s"
        )
        self.mu_s = model.Parameter(name, "mu_s", 0.9, min=0.0, max=1.0)
        self.log10_r = model.Parameter(name, "log10_r", 17.0, min=16.0, max=20.0)
        # disk parameters
        self.log10_L_disk = model.Parameter(
            name, "log10_L_disk", 45.0, min=42.0, max=48.0
        )
        self.log10_M_BH = model.Parameter(name, "log10_M_BH", 42, min=32, max=45)
        self.m_dot = model.Parameter(
            name, "m_dot", 1e26, min=1e24, max=1e30, units="g s-1"
        )
        self.R_in = model.Parameter(name, "R_in", 1e14, min=1e12, max=1e16, units="cm")
        self.R_out = model.Parameter(
            name, "R_out", 1e17, min=1e12, max=1e19, units="cm"
        )
        # DT parameters
        self.xi_dt = model.Parameter(name, "xi_dt", 0.6, min=0.0, max=1.0)
        self.T_dt = model.Parameter(
            name, "T_dt", 1.0e3, min=1.0e2, max=1.0e4, units="K"
        )
        self.R_dt = model.Parameter(
            name, "R_dt", 2.5e18, min=1.0e17, max=1.0e19, units="cm"
        )

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
                self.t_var,
                self.mu_s,
                self.log10_r,
                self.log10_L_disk,
                self.log10_M_BH,
                self.m_dot,
                self.R_in,
                self.R_out,
                self.xi_dt,
                self.T_dt,
                self.R_dt,
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
            t_var,
            mu_s,
            log10_r,
            log10_L_disk,
            log10_M_BH,
            m_dot,
            R_in,
            R_out,
            xi_dt,
            T_dt,
            R_dt,
        ) = pars
        # add units, scale quantities
        x *= u.Hz
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = c.to_value("cm s-1") * t_var * delta_D / (1 + z) * u.cm
        r = 10 ** log10_r * u.cm
        d_L *= u.cm
        L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
        M_BH = 10 ** log10_M_BH * u.Unit("g")
        m_dot *= u.Unit("g s-1")
        R_in *= u.cm
        R_out *= u.cm
        R_dt *= u.cm
        T_dt *= u.K
        eps_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        # non-thermal components
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
            ssa=True,
            gamma=gamma_to_integrate,
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
            ssa=True,
            gamma=gamma_to_integrate,
        )
        sed_ec_dt = ExternalCompton.evaluate_sed_flux_dt(
            x,
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
            x, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
            x, z, xi_dt * L_disk, T_dt, R_dt, d_L
        )
        return sed_synch + sed_ssc + sed_ec_dt + sed_bb_disk + sed_bb_dt


logging.info("reading PKS1510-089 SED from agnpy datas")
sed_path = pkg_resources.resource_filename(
    "agnpy", "data/mwl_seds/PKS1510-089_2015b.ecsv"
)
sed_table = Table.read(sed_path)
x = sed_table["e_ref"].to("Hz", equivalencies=u.spectral())
y = sed_table["e2dnde"]
y_err_stat = sed_table["e2dnde_errn"]
# array of systematic errors, will just be summed in quadrature to the statistical error
# we assume
# - 30% on VHE gamma-ray instruments
# - 10% on HE gamma-ray instruments
# - 10% on X-ray instruments
# - 5% on lower-energy instruments
y_err_syst = np.zeros(len(x))
# define energy ranges
nu_vhe = (100 * u.GeV).to("Hz", equivalencies=u.spectral())
nu_he = (0.1 * u.GeV).to("Hz", equivalencies=u.spectral())
nu_x_ray_max = (300 * u.keV).to("Hz", equivalencies=u.spectral())
nu_x_ray_min = (0.3 * u.keV).to("Hz", equivalencies=u.spectral())
vhe_gamma = x >= nu_vhe
he_gamma = (x >= nu_he) * (x < nu_vhe)
x_ray = (x >= nu_x_ray_min) * (x < nu_x_ray_max)
uv_to_radio = x < nu_x_ray_min
# declare systematics
y_err_syst[vhe_gamma] = 0.30
y_err_syst[he_gamma] = 0.10
y_err_syst[x_ray] = 0.10
y_err_syst[uv_to_radio] = 0.05
y_err_syst = y * y_err_syst
# define the data1D object containing it
sed = data.Data1D("sed", x, y, staterror=y_err_stat, syserror=y_err_syst)

# declare a model
agnpy_ec = AgnpyEC()
# global parameters of the blob and the DT
z = 0.361
d_L = Distance(z=z).to("cm")
# blob
Gamma = 20
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
T_dt = 1e3 * u.K
R_dt = 6.47 * 1e18 * u.cm
# size and location of the emission region
t_var = 0.5 * u.d
r = 6e17 * u.cm
# instance of the model wrapping angpy functionalities
# - AGN parameters
# -- distances
agnpy_ec.z = z
agnpy_ec.z.freeze()
agnpy_ec.d_L = d_L.cgs.value
agnpy_ec.d_L.freeze()
# -- SS disk
agnpy_ec.log10_L_disk = np.log10(L_disk.to_value("erg s-1"))
agnpy_ec.log10_L_disk.freeze()
agnpy_ec.log10_M_BH = np.log10(M_BH.to_value("g"))
agnpy_ec.log10_M_BH.freeze()
agnpy_ec.m_dot = m_dot.to_value("g s-1")
agnpy_ec.m_dot.freeze()
agnpy_ec.R_in = R_in.to_value("cm")
agnpy_ec.R_in.freeze()
agnpy_ec.R_out = R_out.to_value("cm")
agnpy_ec.R_out.freeze()
# -- Dust Torus
agnpy_ec.xi_dt = xi_dt
agnpy_ec.xi_dt.freeze()
agnpy_ec.T_dt = T_dt.to_value("K")
agnpy_ec.T_dt.freeze()
agnpy_ec.R_dt = R_dt.to_value("cm")
agnpy_ec.R_dt.freeze()
# - blob parameters
agnpy_ec.delta_D = delta_D
agnpy_ec.delta_D.freeze()
agnpy_ec.log10_B = np.log10(B.to_value("G"))
agnpy_ec.mu_s = mu_s
agnpy_ec.mu_s.freeze()
agnpy_ec.t_var = (t_var).to_value("s")
agnpy_ec.t_var.freeze()
agnpy_ec.log10_r = np.log10(r.to_value("cm"))
agnpy_ec.log10_r.freeze()
# - EED
agnpy_ec.log10_k_e = np.log10(0.05)
agnpy_ec.p1 = 1.8
agnpy_ec.p2 = 3.5
agnpy_ec.log10_gamma_b = np.log10(500)
agnpy_ec.log10_gamma_min = np.log10(1)
agnpy_ec.log10_gamma_min.freeze()
agnpy_ec.log10_gamma_max = np.log10(3e4)
agnpy_ec.log10_gamma_max.freeze()

logging.info("performing the fit")
# directory to store the checks performed on the fit
fit_check_dir = "figures/figure_7_checks_sherpa_fit"
Path(fit_check_dir).mkdir(parents=True, exist_ok=True)
# fit using the Levenberg-Marquardt optimiser
fitter = Fit(sed, agnpy_ec, stat=Chi2(), method=LevMar())
min_x = 1e11 * u.Hz
max_x = 1e30 * u.Hz
sed.notice(min_x, max_x)

results = time_function_call(fitter.fit)
print("fit succesful?", results.succeeded)
print(results.format())
# plot final model without components
nu = np.logspace(10, 30, 300)
plt.errorbar(sed.x, sed.y, yerr=sed.get_error(), marker=".", ls="")
plt.loglog(nu, agnpy_ec(nu))
plt.xlabel(sed_x_label)
plt.ylabel(sed_y_label)
plt.savefig(f"{fit_check_dir}/best_fit.png")
plt.close()

logging.info("plot the final model with the individual components")
# plot the best fit model with the individual components
k_e = 10 ** agnpy_ec.log10_k_e.val * u.Unit("cm-3")
p1 = agnpy_ec.p1.val
p2 = agnpy_ec.p2.val
gamma_b = 10 ** agnpy_ec.log10_gamma_b.val
gamma_min = 10 ** agnpy_ec.log10_gamma_min.val
gamma_max = 10 ** agnpy_ec.log10_gamma_max.val
B = 10 ** agnpy_ec.log10_B.val * u.G
r = 10 ** agnpy_ec.log10_r.val * u.cm
delta_D = agnpy_ec.delta_D.val
R_b = c.to_value("cm s-1") * agnpy_ec.t_var.val * delta_D / (1 + z) * u.cm
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
print(blob)
print(f"jet power in particles: {blob.P_jet_e:.2e}")
print(f"jet power in B: {blob.P_jet_B:.2e}")

# Disk and DT definition
L_disk = 10 ** agnpy_ec.log10_L_disk.val * u.Unit("erg s-1")
M_BH = 10 ** agnpy_ec.log10_M_BH.val * u.Unit("g")
m_dot = agnpy_ec.m_dot.val * u.Unit("g s-1")
eta = (L_disk / (m_dot * c ** 2)).to_value("")
R_in = agnpy_ec.R_in.val * u.cm
R_out = agnpy_ec.R_out.val * u.cm
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
    ec_dt_sed,
    ls="--",
    lw=1.3,
    color="dodgerblue",
    label="agnpy, EC on DT",
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
    nu / (1 + z), ssc_sed, ls="--", lw=1.3, color="lightseagreen", label="agnpy, SSC"
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
# systematics error in gray
ax.errorbar(
    sed.x, sed.y, yerr=sed.get_syserror(), marker=",", ls="", color="gray",
)
# statistics error in black
ax.errorbar(
    sed.x,
    sed.y,
    yerr=sed.get_staterror(),
    marker=".",
    ls="",
    color="k",
    label="PKS 1510-089, Ahnen et al. (2017), period B",
)
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_xlim([1e9, 1e29])
ax.set_ylim([10 ** (-13.5), 10 ** (-7.5)])
ax.legend(
    loc="upper center", fontsize=10, ncol=2,
)
plt.show()
fig.savefig("figures/figure_7_gammapy_fit.png")
fig.savefig("figures/figure_7_gammapy_fit.pdf")
