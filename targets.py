import numpy as np
import astropy.units as u
from astropy.constants import M_sun
from agnpy.emission_regions import Blob
from agnpy.targets import CMB, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt

# energy density part
M_BH = 1e9 * M_sun
L_disk = 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_tilde_in = 6
R_tilde_out = 50

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
print(blob)
# add CMB and the photon density of the synchrotron radiation for comparison
cmb = CMB(z=blob.z)

disk = SSDisk(M_BH, L_disk, eta, R_tilde_in, R_tilde_out, R_g_units=True)
blr = SphericalShellBLR(L_disk, 0.1, "Lyalpha", 1e17 * u.cm)
dt = RingDustTorus(L_disk, 0.2, 1000 * u.K)

r = np.logspace(15, 21) * u.cm

u_cmb = cmb.u(blob)
u_disk = disk.u(r, blob)
u_blr = blr.u(r, blob)
u_dt = dt.u(r, blob)

# plotting
load_mpl_rc()
fig, ax = plt.subplots()
u_label = r"$u'\,/\,{\rm erg}\,{\rm cm}^{-3}$"
r_label = r"$r\,/\,{\rm cm}$"
ax.axhline(u_cmb.to_value("erg cm-3"), lw=2, ls="--", color="k", label="CMB")
ax.loglog(r, u_disk, lw=2, color="crimson", label="Disk")
ax.loglog(r, u_blr, lw=2, color="dodgerblue", label="Broad Line Region")
ax.loglog(r, u_dt, lw=2, color="goldenrod", label="Dust Torus")
ax.axhline(
    blob.u_ph_synch.to_value("erg cm-3"),
    lw=2,
    ls="--",
    color="darkgray",
    label="synchrotron",
)
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
ax.loglog(nu, tau_disk, lw=2, label="Disk")
ax.loglog(nu, tau_blr, lw=2, label="Broad Line Region")
ax.loglog(nu, tau_dt, lw=2, label="Dust Torus")
ax.legend(fontsize=12)
ax.set_xlabel(r"$\nu\,/\,Hz$", fontsize=12)
ax.set_ylabel(r"$\tau_{\gamma \gamma}$", fontsize=12)
ax.grid(ls=":")
plt.show()
fig.savefig("figures/tau_targets.pdf")
fig.savefig("figures/tau_targets.png")
