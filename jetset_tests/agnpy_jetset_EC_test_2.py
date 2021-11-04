from jetset.jet_model import Jet
import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.emission_regions import Blob
from agnpy.targets import SphericalShellBLR
from agnpy.compton import ExternalCompton
from agnpy.utils.plot import sed_x_label, sed_y_label
import matplotlib.pyplot as plt

# agnpy
# - blob
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
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
blob.set_delta_D(Gamma=Gamma, theta_s=0.05 * u.deg)

# - BLR
L_disk = 2 * 1e46 * u.Unit("erg s-1")
xi_line = 0.024
R_line = 1.1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

# jet(set) with broken power-law electron distribution
jet = Jet(
    name="test",
    electron_distribution="bkn",
    electron_distribution_log_values=False,
    beaming_expr="bulk_theta",
)

jet.add_EC_component(["EC_BLR"],disk_type='BB')
# - blob
jet.set_par("N", val=blob.n_e_tot.value)
jet.set_par("p", val=blob.n_e.p1)
jet.set_par("p_1", val=blob.n_e.p2)
jet.set_par("gamma_break", val=blob.n_e.gamma_b)
jet.set_par("gmin", val=blob.n_e.gamma_min)
jet.set_par("gmax", val=blob.n_e.gamma_max)
jet.set_par("R", val=blob.R_b.value)
jet.set_par("B", val=blob.B.value)
jet.set_par("BulkFactor", val=blob.Gamma)
jet.set_par("theta", val=blob.theta_s.value)
jet.set_par("z_cosm", val=blob.z)
# - disk
jet.set_par("L_Disk", val=L_disk.value)
jet.set_par("tau_BLR", val=blr.xi_line)
jet.set_par("R_BLR_out", val=blr.R_line.value)
jet.set_par("R_BLR_in", val=blr.R_line.value*0.8)  # very thin BLR
jet.electron_distribution.update()
jet._blob.theta_n_int=500
print('theta_n_int',jet._blob.theta_n_int)
# compare the two electron distributions
#gamma_jetset = jet.electron_distribution.gamma_e
#n_e_jetset = jet.electron_distribution.n_gamma_e
#plt.loglog(blob.gamma, blob.n_e(blob.gamma), color="crimson", label="agnpy")
#plt.loglog(gamma_jetset, n_e_jetset, ls="--", color="k", label="jetset")
#plt.legend()
#plt.xlabel(r"$\gamma'$")
#plt.ylabel(r"$n'_{\rm e}$")
#plt.show()

# set the same grid in frequency
nu = np.logspace(14, 30, 100) * u.Hz
jet.set_nu_grid(1e14, 1e30, 100)



# compare for different distances
_v=[0.1,1.5,20,100]


c_f=np.zeros(len(_v))
ratios=np.zeros(len(_v))

transformation="disk"
tetha_deg_v=np.array([2,3,20,30])
for i, theta_deg in enumerate(tetha_deg_v):
    fig, ax = plt.subplots(
    1, len(_v), figsize=(12, 4), sharex=False, sharey=True, tight_layout=True
)
    for j, _r in enumerate(_v):
        blob.set_delta_D(Gamma=Gamma, theta_s=theta_deg * u.deg)
        jet.set_par("theta", val=blob.theta_s.value)

        r = _r * blr.R_line
        # - agnpy EC
        ec = ExternalCompton(blob, blr, r=r)
        ec_sed_agnpy = ec.sed_flux(nu)
        # - jetset EC
        jet.set_par("R_H", val=r.to_value("cm"))
        jet.set_external_field_transf(transformation)
        #jet.show_model()
        # evaluate and fetch the EC on disk component
        jet._blob.R_H_scale_factor=max(50,r.to_value("cm")/blr.R_line.to_value("cm"))
        jet._blob.R_H_scale_factor=min(50,jet._blob.R_H_scale_factor)
        jet._blob.R_H_lim=0.5
        jet._blob.theta_lim=5
        jet.eval()
        # fetch nu and nuFnu from jetset
        nu_jetset = jet.spectral_components.EC_BLR.SED.nu
        ec_sed_jetset = jet.spectral_components.EC_BLR.SED.nuFnu
        # eliminate extremly low values
        null_values = ec_sed_jetset.value < 1e-50
        
        ax[j].loglog(nu, ec_sed_agnpy, color="crimson", label="agnpy")
       
        ratios[j]=ec_sed_agnpy.max().value/ec_sed_jetset.max().value
        c_f[j]=jet.get_beaming()/jet.parameters.BulkFactor.val
        mu_s=np.cos(np.arctan(1/_r))
        mu_star=np.sin(np.deg2rad(theta_deg))
        cos_phi_bar=mu_s*mu_star+(np.sqrt(1-mu_s**2)*np.sqrt(1-mu_star**2))
        print('theta',theta_deg,'r',_r,'cos_phi_bar',cos_phi_bar,'ratio',ratios[j],ratios[j]/cos_phi_bar,'delta/Gamma',c_f[j],'R_H_scale_factor',jet._blob.R_H_scale_factor)
        
        ax[j].loglog(
            nu_jetset[~null_values],
            ec_sed_jetset[~null_values]*1,
            ls="--",
            color="k",
            label="jetset",
        )
        ax[j].legend()
        if j==0:
            y_max=ec_sed_jetset[~null_values].max().value*10
            
        if j== len(_v)-1:
            y_min=ec_sed_jetset[~null_values].min().value/1000
            ax[j].set_ylim([y_min, y_max])
        text = (
            f"frame = {transformation}\n" + r"$r = $" + f"{_r}" + r"$\times R_{\rm DT}$ $\theta=%$" + f"{theta_deg}"
        )
        ax[j].text(1e20,y_max, text, bbox=dict(boxstyle="round", fc="w", alpha=0.5))
        ax[j].grid(ls=":")
        ax[j].set_xlabel(sed_x_label)
        ax[j].set_xlabel(sed_y_label)
        fig.suptitle("EC on DT")
        #fig.savefig(f"jetset_ec_dt_comparisons.png")
        #fig.savefig(f"jetset_ec_dt_comparisons.pdf")
        plt.tight_layout()
# set labels


fig = plt.figure()
plt.semilogx(_v, ratios, marker="o")
plt.show()