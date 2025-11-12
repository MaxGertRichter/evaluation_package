import numpy as np

def average_light_level(yaml_config: dict, data: np.ndarray) -> float:
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
        lightlevel = np.average(data[0].flatten())/(number_meas/2)
    elif averaging_mode == "spread":
        lightlevel = np.average(data[0].flatten())/(av)
    return lightlevel


def ref_mess_voltage(yaml_config:dict, data: np.ndarray) -> tuple[float, float]:
    """Calcualtes the voltage level of the reference and measurement signal.

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
    ref = data[0].flatten()
    mess = data[1].flatten()
    volts = np.empty(2,len(ref))

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
    """Calculates the contrast for all measurement types only ESR has a different calculation

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
    if experiment_type == "ESR":
        ref_ESR = data[0].flatten()
        meas_ESR = data[1].flatten()
        contrast = meas_ESR/ref_ESR
    else:
        ref = data[0]
        mess = data[1]
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