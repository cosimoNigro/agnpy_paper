# quick utils functions for the scripts in agnpy_paper 
import time
import logging


logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def time_function_call(func, *args):
    """Execute a function call, time it and return the normal output expected
    from the function."""
    t_start = time.perf_counter()
    val = func(*args)
    t_stop = time.perf_counter()
    delta_t = t_stop - t_start 
    logging.info(f"elapsed time {func} call: {delta_t:.3f} s")
    return val