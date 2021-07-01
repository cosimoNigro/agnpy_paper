# quick utils functions for the scripts in agnpy_paper
import numpy as np
import astropy.units as u
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
    # the first argument is either an array of frequencies
    logging.info(f"elapsed time {func} call: {delta_t:.3f} s")
    # if the first argument is an array of quantities
    if isinstance(args[0], u.Quantity) and isinstance(args[0], np.ndarray):
        logging.info(f"computed over a grid of {len(args[0])} points")
    return val


def reproduce_sed(dataset, process, nu_range):
    """function to reproduce the SED data in a given reference dataset"""
    # reference SED
    sed_data = np.loadtxt(dataset, delimiter=",")
    nu_ref = sed_data[:, 0] * u.Hz
    # apply the comparison range
    comparison = (nu_ref >= nu_range[0]) * (nu_ref <= nu_range[-1])
    nu_ref = nu_ref[comparison]
    sed_ref = sed_data[:, 1][comparison] * u.Unit("erg cm-2 s-1")
    # compute the sed with agnpy on the same frequencies, time it also
    sed_agnpy = time_function_call(process.sed_flux, nu_ref)
    return nu_ref, sed_ref, sed_agnpy
