import numpy as np
from evaluation_package.config import config

# Set configuration variables once when the module loads
_cfg = config._cfg
REF_IDX = _cfg.get("data_channels", {}).get("reference", 0)
MEAS_IDX = _cfg.get("data_channels", {}).get("measurement", 1)

def calc_contrast(data: np.ndarray)-> np.ndarray:
    """Calculates the contrast for the DAQ Read Sweep experiment.

    Parameters
    ----------
    data : np.ndarray
        Experimental data array, index 0 is reference and index one is the measurement.

    Returns
    -------
    np.ndarray
        Contrast array calculated as measurement/reference.
    """
    ref = data[REF_IDX].flatten()
    mess = data[MEAS_IDX].flatten()
    contrast = (ref - mess)/ref
    return contrast

def calc_delay_times(yaml_config: dict) -> np.ndarray:
    """Calculates the delay array fot the daq read sweep evaluation

    Parameters
    ----------
    yaml_config : dict
        Configuration file of the experiment

    Returns
    -------
    np.ndarray
        Delay times array of the relative delay of the DAQ read trigger to the laser pulse.
    """
    max_delay = yaml_config["pulse_sequence"]["max_delay"]
    number_delays = yaml_config["pulse_sequence"]["n_meas"]//2
    delay_times = np.linspace(0, max_delay, number_delays)
    return delay_times

def calc_snr(yaml_config: dict, data: np.ndarray) -> np.ndarray:
    """Calculates the SNR from the experimental data.

    Parameters
    ----------
    data : np.ndarray
        Experimental data array, index 0 is reference and index one is the measurement.
    yaml_config : dict
        Configuration file of the experiment

    Returns
    -------
    np.ndarray
        SNR array calculated as contrast/sqrt(reference/averages)
    """
    contrast = calc_contrast(data)
    snr_psn = np.sqrt(np.abs(data[REF_IDX].flatten())/yaml_config["averages"])
    snr = contrast*snr_psn
    return snr

def find_optimal_delay(yaml_config:dict, data: np.ndarray) -> float:
    """Finds the optimal delay time for the DAQ read sweep experiment.

    Parameters
    ----------
    yaml_config : dict
        Configuration file of the experiment
    data : np.ndarray
        Experimental data array, index 0 is reference and index one is the measurement.

    Returns
    -------
    float
        Optimal delay time in microseconds.
    """
    snr = calc_snr(yaml_config, data)
    delay_times = calc_delay_times(yaml_config)
    optimal_delay = delay_times[np.argmax(snr)]
    return optimal_delay

def laser_starting_time(yaml_config: dict, data: np.ndarray) -> float:
    """
    Calculate the starting time of the laser based on the rise of the DAQ analog input signal.

    This function determines the time at which the laser starts by analyzing the 
    slope of the DAQ signal. It identifies the point of maximum slope in the signal, 
    which corresponds to the steepest rise, and calculates the starting time by 
    tangential extrapolating back to the time when the signal would have been zero.

    Args:
        yaml_config (dict): Configuration dictionary containing the DAQ settings, 
            including delay times.
        data (np.ndarray): 2D array of DAQ signal data. The first row is used to 
            calculate the rise of the signal.

    Returns:
        float: The calculated starting time of the laser in the same time units 
        as the delay times.
    """
    delay_times = calc_delay_times(yaml_config)
    dt = delay_times[1] - delay_times[0]
    daq_rise = data[0].flatten()
    slope = np.gradient(daq_rise, dt)
    max_slope = np.max(slope)
    max_slope_index = np.argmax(slope)
    time_max_slope = delay_times[max_slope_index]
    t_0 = time_max_slope - (daq_rise[max_slope_index] / max_slope)
    return t_0