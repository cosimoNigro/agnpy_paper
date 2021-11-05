import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.targets import lines_dictionary, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt
from pathlib import Path
from utils import time_function_call

# distance of the source
z = 1
# dictionary of the shells we want to use
ly_alpha_line = lines_dictionary["Lyalpha"]
H_alpha_line = lines_dictionary["Halpha"]
# build a BLR composed of two spherical shells emitting two different lines: Ly alpha and H alpha
L_disk = 2 * 1e46 * u.Unit("erg s-1")
# the luminosity of the shell formed only by the Lyman Alpha line is a fraction 0.1 of the disk luminosity
xi_ly_alpha = 0.024
L_ly_alpha = xi_ly_alpha * L_disk
R_ly_alpha = 1.1e17 * u.cm
# all the shells radiuses and luminosities are given as a function of the Lybeta shell
# obtain its radius and luminosity from the Ly alpha dictionary
R_ly_beta = R_ly_alpha / ly_alpha_line["R_Hbeta_ratio"]
L_ly_beta = L_ly_alpha / ly_alpha_line["L_Hbeta_ratio"]
# obtain H alpha radius and luminosity from the
L_H_alpha = L_ly_beta * H_alpha_line["L_Hbeta_ratio"]
R_H_alpha = R_ly_beta * H_alpha_line["R_Hbeta_ratio"]
# spherical shells emitting the single lines
blr_ly_alpha = SphericalShellBLR(
    L_disk, (L_ly_alpha / L_disk).to_value(""), "Lyalpha", R_ly_alpha
)
blr_H_alpha = SphericalShellBLR(
    L_disk, (L_H_alpha / L_disk).to_value(""), "Halpha", R_H_alpha
)
# dust torus
dt = RingDustTorus(L_disk, 0.1, 1000 * u.K)
# print targets to check their parameters
print(blr_ly_alpha)
print(blr_H_alpha)
print(dt)
# distance from the central sources
r = 1e16 * u.cm

# add absorption on synchrotron photons
# blob definition
spectrum_norm = 6e42 * u.erg
parameters = {
    "p1": 2.0,
    "p2": 3.5,
    "gamma_b": 1e4,
    "gamma_min": 20,
    "gamma_max": 5e7,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# absorptions
abs_blr_ly_alpha = Absorption(blr_ly_alpha, r=r, z=z)
abs_blr_H_alpha = Absorption(blr_H_alpha, r=r, z=z)
abs_dt = Absorption(dt, r=r, z=z)
abs_synch = Absorption(blob)
# array of energies to compute the absorption
nu = np.logspace(24, 30, 100) * u.Hz
# opacities
tau_blr_ly_alpha = time_function_call(abs_blr_ly_alpha.tau, nu)
tau_blr_H_alpha = time_function_call(abs_blr_H_alpha.tau, nu)
tau_dt = time_function_call(abs_dt.tau, nu)
tau_synch = time_function_call(abs_synch.tau, nu)
tau_ext = tau_blr_ly_alpha + tau_blr_H_alpha + tau_dt

# plot
load_mpl_rc()
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.loglog(
    nu,
    tau_blr_ly_alpha,
    lw=2,
    ls="--",
    label="BLR, " + r"${\rm Ly\,\alpha}$" + " shell",
    color="crimson",
)
ax.loglog(
    nu,
    tau_blr_H_alpha,
    lw=2,
    ls="--",
    label="BLR, " + r"${\rm H\,\alpha}$" + " shell",
    color="dodgerblue",
)
ax.loglog(nu, tau_dt, lw=2, ls="--", label="DT", color="goldenrod")
ax.loglog(nu, tau_ext, lw=2, ls="-", label="external", color="k")
ax.fill_between(nu, np.zeros(len(nu)), tau_ext, alpha=0.6, color="darkgray")
ax.loglog(
    nu,
    1e3 * tau_synch,
    lw=2,
    color="lightseagreen",
    label=r"$10^3 \times$" + " synchrotron",
)
ax.fill_between(
    nu, np.zeros(len(nu)), 1e3 * tau_synch, alpha=0.6, color="lightseagreen"
)
ax.legend(loc="best", fontsize=11)
ax.set_xlabel(r"$\nu\,/\,Hz$")
ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
ax.set_ylim([1e-1, 1e3])
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/figure_4.pdf")
fig.savefig("figures/figure_4.png")
