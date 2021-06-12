import numpy as np
import astropy.units as u
from agnpy.targets import lines_dictionary, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt

# distance of the source
z = 1
# dictionary of the shells we want to use
ly_alpha_line = lines_dictionary["Lyalpha"]
H_alpha_line = lines_dictionary["Halpha"]
# build a BLR composed of two spherical shells emitting two different lines: Ly alpha and H alpha
L_disk = 2 * 1e46 * u.Unit("erg s-1")
# the luminosity of the shell formed only by the Lyman Alpha line is a fraction 0.1 of the disk luminosity
xi_ly_alpha = 0.1
L_ly_alpha = 0.1 * L_disk
R_ly_alpha = 1.1 * 1e17 * u.cm
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
# print the BLRs
print(blr_ly_alpha)
print(blr_H_alpha)
# dust torus
dt = RingDustTorus(L_disk, 0.2, 1000 * u.K)
# distance from the central sources
r = 1.1e16 * u.cm

# absorptions
abs_blr_ly_alpha = Absorption(blr_ly_alpha, r=r, z=z)
abs_blr_H_alpha = Absorption(blr_H_alpha, r=r, z=z)
abs_dt = Absorption(dt, r=r, z=z)
# array of energies to compute the absorption
E = np.logspace(0, 5) * u.GeV
nu = E.to("Hz", equivalencies=u.spectral())
# opacities
tau_blr_ly_alpha = abs_blr_ly_alpha.tau(nu)
tau_blr_H_alpha = abs_blr_H_alpha.tau(nu)
tau_dt = abs_dt.tau(nu)
total_tau = tau_blr_ly_alpha + tau_blr_H_alpha + tau_dt

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
ax.loglog(nu, total_tau, lw=2, ls="-", label="total", color="k")
ax.fill_between(nu, np.zeros(len(nu)), total_tau, alpha=0.5, color="darkgray", zorder=1)
ax.legend(loc="best")
ax.set_xlabel(r"$\nu\,/\,Hz$")
ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
fig.savefig("figures/figure_5.pdf")
fig.savefig("figures/figure_5.png")
