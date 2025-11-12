from matplotlib import pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.fft import fft, rfft, fftfreq, rfftfreq
from evaluation_package import utils as ut


#-----------General CASR functions----------------

def calc_calibration_frequency(yaml_config: dict) -> float:
    """Calculate the calibration frequency based on the pulse sequence parameters.

    Args:
        yaml_config (dict): The configuration dictionary containing pulse sequence parameters.

    Returns:
        float: The calculated calibration frequency in Hz.
    """
    tau = yaml_config["pulse_sequence"]["tau"] * 1e-6  # Convert microseconds to seconds
    rf_freq = yaml_config["static_devices"]["rf_source"]["config"]["frequency"]
    sensing_freq = 1 / (4 * tau)
    calibration_freq = np.abs(rf_freq - sensing_freq)
    return calibration_freq


def calc_adjusted_samples(yaml_config: dict) -> tuple[int, float]:
    """Calculates the adjusted number of measurements for a CASR meausrement to fit a integer 
    number of calibration signals in the measurement block.

    Parameters
    ----------
    yaml_config : dict
        The configuration yaml_file to configrue the CASR experiment.

    Returns
    -------
    tuple[int, float]
        The adjusted number of samples and the number of calibration periodes that fit in the measurement block.
    """
    frequency = calc_calibration_frequency(yaml_config)
    delta_t = pulse_sequence_duration(yaml_config)  # Convert microseconds to seconds
    N_initial = yaml_config["pulse_sequence"]["n_meas"] // 2  # Initial number of samples (half of n_meas)
    period = 1 / frequency
    
    # Calculate the total time for the initial number of samples
    total_time = N_initial * delta_t
    
    # Find the closest total time to an integer number of cycles
    closest_total_time = round(total_time / period) * period
    
    # Calculate the adjusted number of samples
    adjusted_N_initial = round(closest_total_time / delta_t)
    N_cycles = closest_total_time / period
    return adjusted_N_initial, N_cycles



def calc_fourier_frequencies(yaml_config: dict) -> np.ndarray:
    """Calculates the fourier frequencies spacing for the CASR experiment.

    Parameters
    ----------
    cfg : dict
        The configuration yaml_file to configrue the experiment.

    Returns
    -------
    np.ndarray
        The fourier frequency axis
    """
    # this needs to be implemented in the other cases also
    adjusted_samples = calc_adjusted_samples(yaml_config)[0]
    dt = pulse_sequence_duration(yaml_config)
    #samples = yaml_config["pulse_sequence"]["n_meas"] // 2
    # old number of samples
    return rfftfreq(adjusted_samples, d=dt)

def calc_fourier_transform(yaml_config: dict, data: np.ndarray, values = "abs") -> np.ndarray:
    """Calculates the fourier transform magnitude or complex values of the CASR contrast signal.
    This function uses the adjusted sample length to calculate the fourier transform.

    Parameters
    ----------
    yaml_config : dict
        The configuration yaml_file to configrue the CASR experiment.
    data : np.ndarray
        The measurement data array of the CASR experiment.
        

    Returns
    -------
    np.ndarray
        The fourier transformed contrast signal.
    """
    # check that an integer number of oscillations fit into the window for maximal sensitivity
    adjusted_samples = calc_adjusted_samples(yaml_config)[0]
    contrast = ut.contrast(data)[:adjusted_samples] 

    if values == "abs":
        fft_final = np.abs(rfft(contrast, norm ="forward"))
    elif values == "complex":
        fft_final = rfft(contrast, norm ="forward")
    return fft_final

def pulse_sequence_duration(yaml_config: dict) -> float:
    """Calculates the total duration of the pulse sequence cycle.

    Parameters
    ----------
    yaml_config : dict
        The configuration yaml_file to configrue the CASR experiment.

    Returns
    -------
    float
        The total duration of the pulse sequence cycle in seconds.
    """
    if "duration_pulseseq_cycle" in yaml_config:
        return yaml_config["duration_pulseseq_cycle"] * 1e-6
    elif "duration_puseseq_cycle" in yaml_config:
        return yaml_config["duration_puseseq_cycle"] * 1e-6
    else:
        raise KeyError("Duration of pulse sequence cycle not found in configuration.")


#-----------Sensitivity calculation functions-----------------

def calc_measurement_time(yaml_config: dict) -> float:
    ps_duration = pulse_sequence_duration(yaml_config) # in seconds
    n_meas = yaml_config["sensor"]["config"]["number_measurements"]//2
    measurement_time = ps_duration * n_meas # this must be divided by two???
    return measurement_time

def find_peak_near(x, y, f0, window_hz=None, window_bins=5):
    """
    Find the peak near frequency f0.

    Parameters
    ----------
    x : array-like
        Frequency axis (Hz).
    y : array-like
        Spectrum amplitude (same length as x).
    f0 : float
        Target frequency (Hz).
    window_hz : float, optional
        Half-width of search window in Hz.
    window_bins : int, default=5
        Half-width of search window in bins (used if window_hz is None).

    Returns
    -------
    idx : int
        Index of the peak.
    freq : float
        Frequency at the peak.
    amp : float
        Amplitude at the peak.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    i0 = np.argmin(np.abs(x - f0))
    
    if window_hz is not None:
        mask = (x >= f0 - window_hz) & (x <= f0 + window_hz)
    else:
        L = max(0, i0 - window_bins)
        R = min(len(x), i0 + window_bins + 1)
        mask = np.zeros_like(x, dtype=bool)
        mask[L:R] = True

    idx_local = np.argmax(y[mask])
    idx = np.where(mask)[0][idx_local]

    return idx, x[idx], y[idx]

def noise_only_mask(
    f: np.ndarray,
    psd: np.ndarray,
    *,
    prominence: float = None,
    height: float = None,
    distance_hz: float = None,
    width_hz: float = None,
    rel_pad: float = 0.1,       # extra padding as a fraction of the peak width
    abs_pad_hz: float = 0.0,    # extra absolute padding in Hz
    fmin: float = None,
    fmax: float = None,
):
    """
    Build a boolean mask (True = noise-only) by cutting out peak regions.

    Parameters
    ----------
    f : array
        Frequency axis (Hz), 1D, ascending, uniform spacing preferred.
    psd : array
        Spectrum/PSD/magnitude for corresponding bins (same shape as f).
    prominence, height, distance_hz, width_hz :
        Tuning for peak detection (passed to scipy.signal.find_peaks).
        - distance_hz is converted to 'distance' in samples.
        - width_hz sets a minimum width in samples for peaks.
    rel_pad : float
        Fractional padding added to each side of the detected peak width.
        e.g. 0.1 means 10% of the (left,right) width each side.
    abs_pad_hz : float
        Absolute padding (Hz) added in addition to the relative padding.
    fmin, fmax : float
        Optional band limit; mask is False outside [fmin, fmax].

    Returns
    -------
    mask : boolean array
        True for noise-only bins; False where peaks (plus padding) exist
        or outside [fmin, fmax] if given.
    peaks : dict
        Information on detected peaks: indices, freqs, widths_hz, intervals_hz.
    """
    f = np.asarray(f)
    psd = np.asarray(psd)
    assert f.ndim == 1 and psd.ndim == 1 and f.size == psd.size

    # frequency resolution (assume uniform grid)
    df = np.median(np.diff(f))

    # Convert user-friendly params from Hz to samples
    distance = int(np.ceil(distance_hz / df)) if distance_hz else None
    width_samples = int(np.ceil(width_hz / df)) if width_hz else None

    # 1) Find peaks
    idx, props = find_peaks(psd,
                            prominence=prominence,
                            height=height,
                            distance=distance,
                            width=width_samples)

    # 2) Estimate peak widths at half-prominence (in samples)
    #    Returns (widths, h_eval, left_ips, right_ips) with float indices
    if idx.size > 0:
        widths_s, _, left_ips, right_ips = peak_widths(psd, idx, rel_height=0.5)
    else:
        widths_s = np.array([])
        left_ips = np.array([])
    # 3) Build exclusion intervals per peak, with padding
    intervals = []
    for i, w_s in enumerate(widths_s):
        # base interval in samples
        left = left_ips[i]
        right = right_ips[i]
        # padding in samples: relative to width + absolute in Hz
        pad_s = rel_pad * w_s + (abs_pad_hz / df)
        L = max(0, int(np.floor(left - pad_s)))
        R = min(psd.size - 1, int(np.ceil(right + pad_s)))
        # to Hz interval
        intervals.append((f[L], f[R]))

    # 4) Merge overlapping intervals
    def merge_intervals(ivals):
        if not ivals:
            return []
        ivals = sorted(ivals, key=lambda x: x[0])
        merged = [ivals[0]]
        for s, e in ivals[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1] = (last_s, max(last_e, e))  # extend
            else:
                merged.append((s, e))
        return merged

    merged_intervals = merge_intervals(intervals)

    # 5) Build mask: start with all True (noise), then cut out merged intervals
    mask = np.ones_like(psd, dtype=bool)

    # Optional band-limits
    if fmin is not None:
        mask &= (f >= fmin)
    if fmax is not None:
        mask &= (f <= fmax)

    for (a, b) in merged_intervals:
        L = max(0, int(np.floor((a - f[0]) / df)))
        R = min(psd.size - 1, int(np.ceil((b - f[0]) / df)))
        mask[L:R+1] = False

    peaks_info = {
        "indices": idx,
        "freqs_hz": f[idx] if idx.size else np.array([]),
        "widths_hz": widths_s * df if idx.size else np.array([]),
        "intervals_hz": merged_intervals,
        "prominences": props.get("prominences", None),
        "heights": props.get("peak_heights", None),
    }
    return mask, peaks_info


def calc_sensitivity(yaml_config: dict, data: np.ndarray, **kwargs)-> tuple[float, float, float]:
    """Calculates the sensitivity and SNR of a singleshot CASR measurement.

    Parameters
    ----------
    yaml_config : dict
        Configuration file of the experiment.
    data : np.ndarray
        Measurement data array of the CASR experiment.
    f0 : float, optional
        The offset freqeuncy of the calibration signal, by default 500

    Returns
    -------
    tuple[float, float, float]
        sensitivity in T/√Hz and normalized SNR and std of the noise
    """
    prominence = kwargs.get("prominence", 0.0001)
    rel_pad = kwargs.get("rel_pad", 10)
    mask_index = kwargs.get("mask_index", 20)
    window_hz = kwargs.get("window_hz", 50)
    width_hz = kwargs.get("width_hz", 1)
    window_bins = kwargs.get("window_bins", 5)
    f0 = calc_calibration_frequency(yaml_config)
    

    frequencies = calc_fourier_frequencies(yaml_config)[mask_index:]
    fft_spectrum_abs = calc_fourier_transform(yaml_config, data)[mask_index:]
    idx, freq, amp = find_peak_near(frequencies, fft_spectrum_abs, f0, window_hz=window_hz, window_bins=window_bins)
    measurement_time = calc_measurement_time(yaml_config)
    print("new version of sensitivity calculation")
    #calculate noise std
    noise_mask, peak_info = noise_only_mask(frequencies, fft_spectrum_abs, prominence=prominence, rel_pad=rel_pad, width_hz=width_hz)
    noise_floor = ut.rms(fft_spectrum_abs[noise_mask])
    snr = amp/noise_floor
    #calculate sensitivity
    sensitivity = 10e-9 /snr * np.sqrt(measurement_time) # in T/√Hz

    return sensitivity, snr, noise_floor