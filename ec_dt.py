import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.targets import RingDustTorus
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import load_mpl_rc
import matplotlib.pyplot as plt

load_mpl_rc()

# blob
spectrum_norm = 1e48 * u.Unit("erg")
spectrum_dict = {
    "type": "LogParabola",
    "parameters": {
        "p": 2.3,
        "q": 0.2,
        "gamma_0": 1e3,
        "gamma_min": 10,
        "gamma_max": 1e7,
    },
}
R_b = 1e16 * u.cm
B = 1 * u.G
z = 0.2
delta_D = Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
blob.set_gamma_size(500)
# target
dt = RingDustTorus(2e46 * u.Unit("erg s-1"), 0.2, 1000 * u.K)

nu = np.logspace(15, 28) * u.Hz

# plot
fig, ax = plt.subplots()

for r in [1e19 * u.cm, 1e20 * u.cm, 1e21 * u.cm]:
    ec = ExternalCompton(blob, dt, r)
    sed = ec.sed_flux(nu)
    _pow = str(int(np.log10(r.to_value("cm"))))
    ax.loglog(nu, sed, lw=2, label=r"$r = 10^{" + _pow + r"}\,{\rm cm}$")

sed_x_label = r"$\nu\,/\,Hz$"
sed_y_label = r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"
ax.grid(ls=":")
ax.legend()
ax.set_xlabel(sed_x_label)
ax.set_ylabel(sed_y_label)
plt.show()
fig.savefig("figures/ec_dt.pdf")
