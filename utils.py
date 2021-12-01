# quick utils functions for the scripts in agnpy_paper
import numpy as np
import astropy.units as u
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.absorption import Absorption
import time
import logging


logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def time_function_call(func, *args, **kwargs):
    """Execute a function call, time it and return the normal output expected
    from the function."""

    t_start = time.perf_counter()
    val = func(*args, **kwargs)
    t_stop = time.perf_counter()
    delta_t = t_stop - t_start
    logging.info(f"elapsed time {func.__name__} call: {delta_t:.3f} s")

    return val


def time_sed_flux(process, nu):
    """For a given radiative process, print the integration grid and time the 
    call to the sed_flux function."""

    if isinstance(process, Synchrotron):
        logging.info("timing synchrotron SED calculation")
        logging.info(f"grid with {process.blob.gamma_size} Lorentz factors")
        logging.info(f"computed on {len(nu)} frequency points")
        return time_function_call(process.sed_flux, nu)

    elif isinstance(process, SynchrotronSelfCompton):
        logging.info("timing SSC SED calculation")
        logging.info(f"grid with {process.blob.gamma_size} Lorentz factors")
        logging.info(f"computed on {len(nu)} frequency points")
        return time_function_call(process.sed_flux, nu)

    elif isinstance(process, ExternalCompton):
        logging.info("timing EC SED calculation")
        logging.info(f"EC grid with {process.blob.gamma_size} Lorentz factors")
        logging.info(f"EC grid with {process.mu_size} zenith (mu) points")
        logging.info(f"EC grid with {process.phi_size} azimuths (phi) points")
        logging.info(f"computed on {len(nu)} frequency points")
        return time_function_call(process.sed_flux, nu)

    else:
        raise TypeError(f"{process} is not an agnpy radiative process")


def time_tau(absorption, nu):
    """Same as time_sed_flux, but timing the opacity calculation"""

    if isinstance(absorption, Absorption):
        logging.info("timing absorption calculation")
        logging.info(f"Abs. grid with {absorption.mu_size} zenith (mu) points")
        logging.info(f"Abs. grid with {absorption.phi_size} azimuths (phi) points")
        logging.info(f"Abs. grid with {absorption.l_size} distance points")
        logging.info(f"computed on {len(nu)} frequency points")
        return time_function_call(absorption.tau, nu)

    else:
        raise TypeError(f"{absorption} is not an agnpy absorption process")


def reproduce_sed(dataset, process, nu_range):
    """Function to reproduce the SED data in a given reference dataset.
    The execution is also timed.
    """
    # reference SED
    sed_data = np.loadtxt(dataset, delimiter=",")
    nu_ref = sed_data[:, 0] * u.Hz
    # apply the comparison range
    comparison = (nu_ref >= nu_range[0]) * (nu_ref <= nu_range[-1])
    nu_ref = nu_ref[comparison]
    sed_ref = sed_data[:, 1][comparison] * u.Unit("erg cm-2 s-1")
    # compute the sed with agnpy on the same frequencies, time it also
    sed_agnpy = time_sed_flux(process, nu_ref)
    return nu_ref, sed_ref, sed_agnpy
