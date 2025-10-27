import numpy as np
from scipy.signal import savgol_filter
from evaluation_package import utils as ut

def linspace_discrete_by_intervals(length_us: float,
                                   n_intervals: int,
                                   sampling_rat: float,
                                   tol_samples: float = 1e-9):

    """
    Endpoint is always excluded.
    """
    Ts_us = (1.0 / sampling_rat) * 1e6  # sample period in microseconds

    # Number of points implied by "number of spacings"
    n_points = n_intervals +1

    # Intended spacing between points
    dt_us = length_us / n_intervals

    # Samples per spacing
    step_exact = dt_us / Ts_us
    step_rounded = int(round(step_exact))

    if step_rounded < 1:
        raise ValueError(
            f"Requested spacing {dt_us*1000:.3f} ns is < 1 sample ({Ts_us*1000:.3f} ns)."
        )

    # If not aligned, suggest nearest valid n_intervals
    if abs(step_exact - step_rounded) > tol_samples:
        nearest_n_intervals = round(length_us / (step_rounded * Ts_us))
        nearest_spacing_us = step_rounded * Ts_us
        max_interval = step_rounded *Ts_us *nearest_n_intervals
        raise ValueError(
            f"Spacing {dt_us:.9f} µs is not on the sampling grid "
            f"({step_exact:.9f} samples/step, nearest integer {step_rounded}).\n"
            f"Nearest valid n_intervals: {nearest_n_intervals} "
            f"(spacing {nearest_spacing_us:.9f} µs)."
            f"Maximum interval: {max_interval:.9f} µs."
        )

    # Build using integer sample steps
    idx = np.arange(n_points, dtype=np.int64) * step_rounded
    t_us = idx * Ts_us

    return t_us[1:].tolist()



def microwave_list(yaml_config: dict) -> list:
    n_meas = yaml_config["pulse_sequence"]["n_meas"]/2
    sampling_rate = yaml_config["synchroniser"]["config"]["sampling_rate"]
    max_mw_duration = yaml_config["pulse_sequence"]["max_mw_duration"]  
    return linspace_discrete_by_intervals(max_mw_duration, n_meas, sampling_rate)


def pi_pulse_duration(yaml_config: dict, data: np.ndarray, **kwargs) -> float:
    defaults = {
        "window_length": 5,
        "polyorder": 3,
        "savgol_filter": True,
    }
    params = {**defaults, **kwargs}
    y_Rabi = ut.contrast(data)
    mw_list = microwave_list(yaml_config)
    if params["savgol_filter"]:
        y_Rabi = savgol_filter(y_Rabi, params["window_length"], params["polyorder"])
    min_i = np.argmin(y_Rabi)
    
    pi_pulse = mw_list[min_i]
    return pi_pulse