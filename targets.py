import numpy as np
import astropy.units as u
from astropy.constants import M_sun
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt

load_mpl_rc()

# energy density part
M_BH = 1e9 * M_sun
L_disk = 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_tilde_in = 6
R_tilde_out = 50

blob = Blob()

disk = SSDisk(M_BH, L_disk, eta, R_tilde_in, R_tilde_out, R_g_units=True)
blr = SphericalShellBLR(L_disk, 0.1, "Lyalpha", 1e17 * u.cm)
dt = RingDustTorus(L_disk, 0.2, 1000 * u.K)

r = np.logspace(15, 21) * u.cm

u_disk = disk.u(r, blob)
u_blr = blr.u(r, blob)
u_dt = dt.u(r, blob)

fig, ax = plt.subplots()
u_label = r"$u'\,/\,{\rm erg}\,{\rm cm}^{-3}$"
r_label = r"$r\,/\,{\rm cm}$"
ax.loglog(r, u_disk, lw=2, label="Disk")
ax.loglog(r, u_blr, lw=2, label="Broad Line Region")
ax.loglog(r, u_dt, lw=2, label="Dust Torus")
ax.grid(ls=":")
ax.legend(fontsize=12)
ax.set_xlabel(r_label, fontsize=12)
ax.set_ylabel(u_label, fontsize=12)
plt.show()
fig.savefig("figures/u_targets.pdf")
fig.savefig("figures/u_targets.png")

# absorption part
from agnpy.absorption import Absorption

r = 1.1e16 * u.cm

absorption_disk = Absorption(blob, disk, r=r)
absorption_blr = Absorption(blob, blr, r=r)
absorption_dt = Absorption(blob, dt, r=r)

E = np.logspace(0, 5) * u.GeV
nu = E.to("Hz", equivalencies=u.spectral())

tau_disk = absorption_disk.tau(nu)
tau_blr = absorption_blr.tau(nu)
tau_dt = absorption_dt.tau(nu)

fig, ax = plt.subplots()
ax.loglog(nu, tau_disk, lw=2, label = "Disk")
ax.loglog(nu, tau_blr, lw=2, label = "Broad Line Region")
ax.loglog(nu, tau_dt, lw=2, label = "Dust Torus")
ax.legend(fontsize=12)
ax.set_xlabel(r"$\nu\,/\,Hz$", fontsize=12)
ax.set_ylabel(r"$\tau_{\gamma \gamma}$", fontsize=12)
ax.grid(ls=":")
plt.show()
fig.savefig("figures/tau_targets.pdf")
fig.savefig("figures/tau_targets.png")
