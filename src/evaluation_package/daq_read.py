import numpy as np

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
    ref = data[0].flatten()
    mess = data[1:].flatten()
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
    snr_psn = np.sqrt(data[0].flatten()/yaml_config["averages"])
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