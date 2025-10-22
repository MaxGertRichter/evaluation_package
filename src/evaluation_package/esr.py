import numpy as np
from scipy.signal import savgol_filter, find_peaks

def esr_frequencies(yaml_config: dict) -> np.ndarray:
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
    esr_freq = np.linspace(min_freq, max_freq, dyn_steps)
    return esr_freq


def esr_contrast(data: np.ndarray) -> np.ndarray:
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
    esr_contrast = meas_ESR/ref_ESR
    return esr_contrast


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

