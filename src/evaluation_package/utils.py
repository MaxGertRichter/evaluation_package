from typing import Union
import numpy as np
from evaluation_package.config import config

# Set configuration variables once when the module loads
_cfg = config._cfg
REF_IDX = _cfg.get("data_channels", {}).get("reference", 0)
MEAS_IDX = _cfg.get("data_channels", {}).get("measurement", 1)

def average_light_level(yaml_config: dict, data: np.ndarray, reference_channel = 0) -> float:
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
    if averaging_mode == "sum":
        lightlevel = np.average(data[reference_channel].flatten())/(number_meas/2)
    elif averaging_mode == "spread":
        lightlevel = np.average(data[reference_channel].flatten())/(av)
    return lightlevel


def average_channel(data: np.ndarray, reference_channel = 0) -> float:
    """Calculates the average of a channel in the data array.

    Parameters
    ----------
    data : np.ndarray
        Data array of the measurement
    reference_channel : int, optional
        Channel to calculate the average of, by default 0

    Returns
    -------
    float
        Average of the channel
    """
    return np.mean(data[reference_channel,0,:,0])

def ref_mess_voltage(yaml_config:dict, data: np.ndarray) -> np.ndarray:
    """Calculates the voltage level of the reference and measurement signal.

    Parameters
    ----------
    yaml_config : dict
        Yaml configuration file for the experiment run
    data : np.ndarray
        data_array of the measurment

    Returns
    -------
    tuple[float, float]
        Voltage level of the reference and measurement signal
    """
    ref = data[REF_IDX].flatten()
    mess = data[MEAS_IDX].flatten()
    volts = np.empty((2, len(ref)))

    averaging_mode = yaml_config["data"]["averaging_mode"]
    number_meas = yaml_config['sensor']['config']['number_measurements']
    av = yaml_config['averages']
    if averaging_mode == "sum":
        volts[0] = ref/(number_meas/2)
        volts[1] = mess/(number_meas/2)
    elif averaging_mode == "spread":
        volts[0] = ref/(av)
        volts[1] = mess/(av)
    return volts


def contrast(data: np.ndarray, experiment_type = None) -> np.ndarray:
    """Calculates the contrast for all measurement types.

    Parameters
    ----------
    data : np.ndarray
        experimental data array
    experiment_type : _type_, optional
        determines the experiment type and hence which contrast to calculate, by default None

    Returns
    -------
    np.ndarray
        contrast of the measurement
    """
    # Convert lists to a numpy array for universal handling
    if isinstance(data, list):
        data = np.array(data)

    # Autocorrect 5-dimensional data layouts down to 4D
    if data.ndim == 5 and data.shape[2] == 1:
        data = np.squeeze(data, axis=2)

    exp_type = "" if experiment_type is None else str(experiment_type)

    if exp_type in ("ESR", "Rabi"):
        ref_ESR = data[REF_IDX].flatten()
        meas_ESR = data[MEAS_IDX].flatten()
        contrast = meas_ESR/ref_ESR
    elif "CASR" in exp_type:
        ref_CASR = data[REF_IDX].squeeze()
        meas_CASR = data[MEAS_IDX].squeeze()
        # CASR calibration and sensitivity uses ref - meas
        contrast = ref_CASR - meas_CASR
    else:
        ref = data[REF_IDX]
        mess = data[MEAS_IDX]
        contrast = np.squeeze((mess - ref) / (mess + ref))
    
    return contrast


def rms(arr: np.ndarray) -> float:
    """Calculate the root mean square (RMS) of a NumPy array.

    Args:
        arr (np.ndarray): Input array for which to calculate the RMS.

    Returns:
        float: The RMS value of the input array.
    """
    return np.sqrt(np.mean(np.square(np.abs(arr))))

def match_resolution(sampling_rate: float, pulse_length_us: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Adjusts the pulse length to the closest value matching the resolution 
    obtained with the sampling rate.
    
    Args:
        sampling_rate (float): Sampling rate in Hz.
        pulse_length_us (float | np.ndarray): Target pulse length(s) in microseconds.
        
    Returns:
        float | np.ndarray: The matched pulse length(s) in microseconds.
    """
    # Calculate sampling period in microseconds
    # period_us = (1 / sampling_rate) * 1e6
    
    # Or simply: Number of samples = duration_seconds * sampling_rate
    # duration_seconds = pulse_length_us * 1e-6
    n_samples = np.round(pulse_length_us * 1e-6 * sampling_rate)
    
    # Recalculate duration from integer samples
    matched_duration_seconds = n_samples / sampling_rate
    matched_duration_us = matched_duration_seconds * 1e6
    matched_duration_us = np.round(matched_duration_us, 12)
    return matched_duration_us