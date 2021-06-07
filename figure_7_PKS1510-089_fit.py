# import numpy, astropy and matplotlib for basic functionalities
import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_e, c, G, M_sun
from astropy.coordinates import Distance
from pathlib import Path
from astropy.table import Table
import matplotlib.pyplot as plt
import pkg_resources

# import agnpy classes
from agnpy.emission_regions import Blob
from agnpy.spectra import BrokenPowerLaw
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


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_size = 400
gamma_to_integrate = np.logspace(0, 7, gamma_size)


class BrokenPowerLawEC(model.RegriddableModel1D):
    """wrapper of agnpy's synchrotron, SSC and EC classes. A broken power-law is assumed for the electron spectrum."""

    def __init__(self, name="ec"):
        # EED parameters
        self.log10_k_e = model.Parameter(name, "log10_k_e", -1.0, min=-10.0, max=10.0)
        self.p1 = model.Parameter(name, "p1", 2.1, min=1.0, max=5.0)
        self.p2 = model.Parameter(name, "p2", 1.0, min=0.0, max=5.0)
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
        self.delta_D = model.Parameter(name, "delta_D", 10, min=1, max=40)
        self.mu_s = model.Parameter(name, "mu_s", 0.9, min=0.0, max=1.0)
        self.log10_B = model.Parameter(name, "log10_B", 0.0, min=-3.0, max=1.0)
        self.alpha_jet = model.Parameter(name, "alpha_jet", 0.05, min=0.0, max=1.1)
        self.log10_r = model.Parameter(name, "log10_r", 17.0, min=16.0, max=20.0)

        # disk parameters
        self.log10_L_disk = model.Parameter(
            name, "log10_L_disk", 45.0, min=42.0, max=48.0
        )
        self.log10_M_BH = model.Parameter(name, "log10_M_BH", 42, min=32, max=45)
        self.m_dot = model.Parameter(name, "m_dot", 1e26, min=1e24, max=1e30)
        self.R_in = model.Parameter(name, "R_in", 1e14, min=1e12, max=1e16)
        self.R_out = model.Parameter(name, "R_out", 1e17, min=1e12, max=1e19)
        # DT parameters
        self.xi_dt = model.Parameter(name, "xi_dt", 0.6, min=0.0, max=1.0)
        self.T_dt = model.Parameter(name, "T_dt", 1.0e3, min=1.0e2, max=1.0e4)
        self.R_dt = model.Parameter(name, "R_dt", 2.5e18, min=1.0e17, max=1.0e19)

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
                self.mu_s,
                self.log10_B,
                self.alpha_jet,
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
        ) = pars
        # add units, scale quantities
        x *= u.Hz
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        d_L *= u.cm
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


# read the 1D data
sed_path = pkg_resources.resource_filename(
    "agnpy", "data/mwl_seds/PKS1510-089_low.ecsv"
)
sed_table = Table.read(sed_path)
x = sed_table["nu"]
y = sed_table["flux"]
y_err = sed_table["flux_err_lo"]
# remove the points with orders of magnitude smaller error, they are upper limits
UL = y_err < (y * 1e-3)
# add an arbitrary systematic error of 10% on the flux of all points
syst_err = 0.1 * y
# load the SED points in the sherpa data object
sed = data.Data1D("sed", x[~UL], y[~UL], staterror=y_err[~UL], syserror=syst_err[~UL])

# global parameters of the blob and the DT
# galaxy distance
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
# load and set all the blob parameters
model = BrokenPowerLawEC()
# - AGN parameters
# -- distances
model.z = z
model.z.freeze()
model.d_L = d_L.cgs.value
model.d_L.freeze()
# -- SS disk
model.log10_L_disk = np.log10(L_disk.to_value("erg s-1"))
model.log10_L_disk.freeze()
model.log10_M_BH = np.log10(M_BH.to_value("g"))
model.log10_M_BH.freeze()
model.m_dot = m_dot.to_value("g s-1")
model.m_dot.freeze()
model.R_in = R_in.to_value("cm")
model.R_in.freeze()
model.R_out = R_out.to_value("cm")
model.R_out.freeze()
# -- Dust Torus
model.xi_dt = xi_dt
model.xi_dt.freeze()
model.T_dt = T_dt.to_value("K")
model.T_dt.freeze()
model.R_dt = R_dt.to_value("cm")
model.R_dt.freeze()
# - blob parameters
model.delta_D = delta_D
model.delta_D.freeze()
model.mu_s = mu_s
model.mu_s.freeze()
model.alpha_jet = alpha_jet
model.alpha_jet.freeze()
model.log10_r = np.log10(r.to_value("cm"))
model.log10_r.freeze()
model.log10_B = np.log10(B.to_value("G"))
model.log10_B.freeze()
# - EED
model.log10_k_e = np.log10(0.5)
model.p1 = 1.9
model.p2 = 3.5
model.log10_gamma_b = np.log10(130)
model.log10_gamma_min = np.log10(2)
model.log10_gamma_min.freeze()
model.log10_gamma_max = np.log10(2e5)
model.log10_gamma_max.freeze()
print(model)
# plot the starting model
nu = np.logspace(9, 30, 200)
plt.errorbar(sed.x, sed.y, yerr=sed.get_error(), ls="", marker=".", color="k")
plt.loglog(nu, model(nu), color="crimson")
plt.ylim([1e-14, 1e-8])
plt.show()


# fit using the Levenberg-Marquardt optimiser
fitter = Fit(sed, model, stat=Chi2(), method=LevMar())
# use confidence to estimate the errors
fitter.estmethod = Confidence()
fitter.estmethod.parallel = True
min_x = 1e11
max_x = 1e30
sed.notice(min_x, max_x)
print(fitter)

# perform the first fit, we are only varying the spectral parameters
print("-- first iteration with only spectral parameters free")
results_1 = fitter.fit()
print("-- fit succesful?", results_1.succeeded)
print(results_1.format())

# perform the second fit, we are varying also the blob parameters
print("-- second iteration with spectral and blob parameters free")
# model.delta_D.thaw()
model.log10_B.thaw()
model.log10_r.thaw()
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
plt.ylim([1e-14, 1e-8])
plt.show()

# plot the best fit model with the individual components
k_e = 10 ** model.log10_k_e.val * u.Unit("cm-3")
p1 = model.p1.val
p2 = model.p2.val
gamma_b = 10 ** model.log10_gamma_b.val
gamma_min = 10 ** model.log10_gamma_min.val
gamma_max = 10 ** model.log10_gamma_max.val
B = 10 ** model.log10_B.val * u.G
r = 10 ** model.log10_r.val * u.cm
delta_D = model.delta_D.val
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
L_disk = 10 ** model.log10_L_disk.val * u.Unit("erg s-1")
M_BH = 10 ** model.log10_M_BH.val * u.Unit("g")
m_dot = model.m_dot.val * u.Unit("g s-1")
eta = (L_disk / (m_dot * c ** 2)).to_value("")
R_in = model.R_in.val * u.cm
R_out = model.R_out.val * u.cm
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
dt = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)
# print model components
print(blob)
print(disk)
print(dt)

# radiative processes
synch = Synchrotron(blob, ssa=True)
ssc = SynchrotronSelfCompton(blob, synch)
ec_dt = ExternalCompton(blob, dt, r)
# SEDs
nu = np.logspace(9, 30, 200) * u.Hz
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
    sed.x,
    sed.y,
    yerr=sed.get_error(),
    marker=".",
    ls="",
    color="k",
    label="PKS 1510-089, Acciari et al. (2018)",
)
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
ax.set_ylim([1e-14, 1e-8])
ax.legend(
    loc="upper center", fontsize=10, ncol=2,
)
plt.show()
fig.savefig("figures/figure_7.png")
fig.savefig("figures/figure_7.pdf")
