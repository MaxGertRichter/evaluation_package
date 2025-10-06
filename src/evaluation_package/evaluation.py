import numpy as np
from scipy.signal import savgol_filter, find_peaks




#--------------General functions------------------
def get_light_level(yaml_config: dict, data: np.ndarray) -> float:
    """Calculate the average light level in milli Volts based on the provided configuration and data.

    Args:
        yaml_config (dict): A dictionary containing configuration details. 
            Expected keys:
                - 'sensor': A dictionary with a 'config' key containing:
                    - 'number_measurements' (int): The total number of measurements taken by the sensor.
                - 'averages' (float): A scaling factor for averaging.
        data (np.ndarray): A NumPy array containing the measurement data. 
            The first row of the array is used for calculations.

    Returns:
        float: The calculated average light level in milli Volts.
    """
    averaging_mode = yaml_config["data"]["averaging_mode"]
    number_meas = yaml_config['sensor']['config']['number_measurements']
    av = yaml_config['averages']
    read_trigs = data.shape[0]
    if averaging_mode == "sum":
        lightlevel = np.average(data[0].flatten())/(number_meas/2)
    elif averaging_mode == "spread":
        lightlevel = np.average(data[0].flatten())/(av)
    return lightlevel

def calc_ref_mess_voltage(data: np.ndarray,) -> tuple[float, float]:
    """Calcualtes the voltage level of the reference and measurement signal.

    Parameters
    ----------
    data : np.ndarray
        Data array of the measurment

    Returns
    -------
    tuple[float, float]
        Mean voltage of the reference and measurement signal
    """
    ref = data[0].flatten()
    mess = data[1].flatten()
    return np.mean(ref), np.mean(mess)



#--------------ESR----------------------
def get_x_ESR_array(yaml_config: dict) -> np.ndarray:
    """
    Generates an array of frequency values for the ESR spectrum based on the given configuration.

    Args:
        yaml_config (dict): A dictionary containing the configuration parameters. 
            It must include:
            - 'dynamic_devices': A dictionary with 'mw_source' configuration, 
              which contains 'frequency' as a list of two floats [min_freq, max_freq].
            - 'dynamic_steps': An integer specifying the number of steps for the frequency range.

    Returns:
        np.ndarray: A NumPy array containing the frequency values (in GHz) 
        evenly spaced between the minimum and maximum frequencies.
    """
    min_freq = float(yaml_config['dynamic_devices']['mw_source']['config']['frequency'][0]) / 1e9
    max_freq = float(yaml_config['dynamic_devices']['mw_source']['config']['frequency'][1]) / 1e9
    dyn_steps = yaml_config['dynamic_steps']
    x_ESR = np.linspace(min_freq, max_freq, dyn_steps)
    return x_ESR


def get_y_ESR_array(data: np.ndarray) -> np.ndarray:
    """
    Computes the y-values of the ESR (Electron Spin Resonance) spectrum.

    This function takes a 2D numpy array where the first element represents 
    the reference ESR spectrum and the second element represents the measured 
    ESR spectrum. It calculates the ratio of the measured spectrum to the 
    reference spectrum to obtain the y-values of the ESR spectrum.

    Parameters:
    -----------
    data : np.ndarray
        A 2D numpy array where:
        - data[0] is the reference ESR spectrum (1D array).
        - data[1] is the measured ESR spectrum (1D array).

    Returns:
    --------
    np.ndarray
        A 1D numpy array containing the y-values of the ESR spectrum.
    """
    ref_ESR = data[0].flatten()
    meas_ESR = data[1].flatten()
    y_ESR = meas_ESR/ref_ESR
    return y_ESR


def get_min_peaks(x, y, **kwargs):
    defaults = {
        "window_length": 21,
        "polyorder": 3,
        "savgol_filter": True,
        "prominence": 5e-4,
        "return_all": False
    }
    # Override defaults with any provided kwargs
    params = {**defaults, **kwargs}
    if params["savgol_filter"]:
        y = savgol_filter(
            y,
            window_length=params["window_length"],
            polyorder=params["polyorder"]
        )
    peaks, props = find_peaks(-y, prominence=params["prominence"])
    if params["return_all"]:
        return peaks
    else:
        return peaks[0]

def get_SRS_frequency(first_peak: float, mixing_frequency: float) -> float:
    """Calculate the frequency of the signal source when you want to mix it

    Parameters
    ----------
    first_peak : float
        Freqeuncy peak of the first peak in the ESR Spectrum
    mixing_frequency : float
        Freqeuncy to mix from the AWG

    Returns
    -------
    float
        Signal source frequency
    """
    return first_peak - mixing_frequency


#----------Rabi----------------------
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



def get_mw_list(yaml_config: dict) -> list:
    n_meas = yaml_config["pulse_sequence"]["n_meas"]/2
    sampling_rate = yaml_config["synchroniser"]["config"]["sampling_rate"]
    max_mw_duration = yaml_config["pulse_sequence"]["max_mw_duration"]  
    return linspace_discrete_by_intervals(max_mw_duration, n_meas, sampling_rate)


def get_y_contrast(data: np.ndarray) -> np.ndarray:
    ref_rabi = data[0].flatten()
    meas_rabi = data[1].flatten()
    y_Rabi = 1 + (meas_rabi - ref_rabi)/ (meas_rabi + ref_rabi) # Careful i ad one here!
    return y_Rabi


def get_pi_pulse(yaml_config: dict, data: np.ndarray, **kwargs) -> float:
    defaults = {
        "window_length": 5,
        "polyorder": 3,
        "savgol_filter": True,
    }
    params = {**defaults, **kwargs}
    y_Rabi = get_y_contrast(data)
    mw_list = get_mw_list(yaml_config)
    if params["savgol_filter"]:
        y_Rabi = savgol_filter(y_Rabi, params["window_length"], params["polyorder"])
    min_i = np.argmin(y_Rabi)
    
    pi_pulse = mw_list[min_i]
    return pi_pulse
