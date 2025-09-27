from matplotlib import pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks, peak_widths


#-----------General CASR functions----------------

def calc_contrast(data: np.ndarray) -> np.ndarray:
    """Calculates the contrast from the reference and measurement data for the CASR experiment.

    Parameters
    ----------
    data : np.ndarray
        the data array from the experiment the first index is the measurement the second is the reference

    Returns
    -------
    np.ndarray
        The contrast array
    """
    ref = data[0]
    mess = data[1]
    return np.squeeze((mess - ref) / (mess + ref))

def calc_fourier_frequencies(cfg: dict) -> np.ndarray:
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
    sensing_time = cfg["duration_puseseq_cycle"] * 1e-6
    samples = cfg["pulse_sequence"]["n_meas"] // 2
    return np.fft.rfftfreq(samples, d=sensing_time)

def calc_fourier_transform(data: np.ndarray) -> np.ndarray:
    final = np.squeeze(calc_contrast(data)) # maybe move the squeeze some where else in the future
    fft_final = np.abs(np.fft.rfft(final, norm ="ortho"))
    return fft_final



#-----------CASR calibration functions-----------------

def calc_b_ac(yaml_config: dict) -> float:
    """Calculates the magnetic field strength b_ac for the CASR calibration, 
    where the phase accumulation is 2pi.

    Parameters
    ----------
    yaml_config : dict
        The configuration yaml_file to configrue the experiment.
    """
    f_rf = yaml_config['dynamic_devices']['rf_source']['config']['frequency'][0]
    N = yaml_config['pulse_sequence']['N'] * 8 # assuming it is a XY8 block
    g_e = constants.physical_constants['electron g factor'][0] *-1
    mu_B = constants.physical_constants['Bohr magneton'][0]
    hbar = constants.hbar
    b_ac = 2 * np.pi**2 * hbar * f_rf /(g_e * mu_B * N)

    return b_ac

def calc_Vpp_list(yaml_config: dict) -> np.ndarray:
    """Calculate the voltage Vpp spacing for the CASR calibration experiment

    Parameters
    ----------
    yaml_config : dict
        Configuration file of the experiment.

    Returns
    -------
    v_axis : np.ndarray
        The voltage axis for the CASR calibration experiment.
    """
    v_min = yaml_config['dynamic_devices']['rf_source']['config']['amplitude'][0]
    v_max = yaml_config['dynamic_devices']['rf_source']['config']['amplitude'][1]
    v_steps = yaml_config['dynamic_steps']
    v_axis = np.linspace(v_min, v_max, v_steps)
    return v_axis

def _odd_bounded(n, w):
    # make w an odd integer in [3, n-1]
    w = int(w)
    if w % 2 == 0: w += 1
    w = max(3, min(w, n-1 if (n-1)%2 else n-2))
    return w

def find_special_dips(
    y, slow_win=10, slow_poly=2, search_win=5, 
    peak_prom=None, peak_dist=None, return_baseline=False
):
    """
    Find 'notches' (downward dips) near each *major* maximum and minimum
    of the underlying big oscillation.

    y : 1D array
    slow_win, slow_poly : Savitzky-Golay smoothing for the big oscillation
    search_win : half-window (in samples) to search around each slow extremum
    peak_prom, peak_dist : passed to find_peaks on the *slow* curve to keep only
                           real maxima/minima of the big oscillation
    return_baseline : if True, also return the smoothed baseline y_slow

    Returns dict with indices:
      - dips_near_max   : notch indices near each slow maximum
      - dips_near_min   : notch indices near each slow minimum
      - slow_maxima/minima : indices of the slow extrema (for reference)
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 7:
        raise ValueError("Trace too short.")

    # 1) slow baseline of the big oscillation
    slow_win = _odd_bounded(n, slow_win)
    y_slow = savgol_filter(y, window_length=slow_win, polyorder=slow_poly)

    # 2) slow maxima & minima (the big oscillation)
    slow_max, _ = find_peaks( y_slow, prominence=peak_prom, distance=peak_dist)
    slow_min, _ = find_peaks(-y_slow, prominence=peak_prom, distance=peak_dist)

    # 3) residual (notch stands out as negative)
    r = y - y_slow

    def best_dip_around(idx: int) -> int:
        L = max(0, idx - search_win)
        R = min(n-1, idx + search_win)
        # choose the *most negative residual* in the neighborhood
        k = L + int(np.argmin(r[L:R+1]))
        return k

    dips_near_max = np.array([best_dip_around(i) for i in slow_max], dtype=int)
    dips_near_min = np.array([best_dip_around(i) for i in slow_min], dtype=int)

    out = dict(
        dips_near_max=dips_near_max,
        dips_near_min=dips_near_min,
        slow_maxima=slow_max,
        slow_minima=slow_min,
    )
    if return_baseline:
        out["baseline"] = y_slow
    return out

def find_backfolding_index(contrast: np.ndarray) -> int:
    """Calculates the index where the backfolding during the CASR calibration happens

    Parameters
    ----------
    contrast : np.ndarray
        the constrast array of the CASR calibration

    Returns
    -------
    backfold_id : int
        The index of a minima in a backfolding maxima
    """
    summed_contrast = np.sum(contrast, axis=0)
    dict_minmax = find_special_dips(summed_contrast)
    backfold_id = dict_minmax["slow_maxima"][0]
    return backfold_id


def b_sine(x, V, O, A, ϕ):
    # fitting fucntion for the CASR calibration
    # V is the voltage periode
    # A is the amplitude
    # O is the offset
    return O + A * np.sin(2*np.pi * x / V + ϕ)



def plot_casr_clibration(yaml_config: dict, data: np.ndarray) -> None:
    """Plots the CASR calibration and fits a sine to the backfolding maxima. Prints the voltage needed to generate a 10 nT signal.

    Parameters
    ----------
    yaml_config : dict
        Configuration file of the experiment.
    data : np.ndarray
        The measurment data array of the CASR calibration.
    """
    contrast = calc_contrast(data)
    backfold_id = find_backfolding_index(contrast)
    maximas_backfolding = contrast[:, backfold_id]
    v_axis = calc_Vpp_list(yaml_config)
    opt, _ = curve_fit(b_sine, v_axis, maximas_backfolding, p0=[v_axis[-1], 0, max(maximas_backfolding), 0])
    b_ac = calc_b_ac(yaml_config)
    C = b_ac/opt[0]
    V_for_10_nT = 10e-9 / C
    print(f'You need {V_for_10_nT:.4f} V to generate a 10 nT Signal')
    plt.plot(v_axis, maximas_backfolding, label="Data")
    plt.plot(v_axis, b_sine(v_axis, *opt), label=f'b_ac = {b_ac:.2e} T\nb/V = {b_ac/opt[0]:.2e} T/V')
    plt.legend()
    plt.xlabel("Vpp (V)")

def check_backfolding_index(contrast: np.ndarray) -> None:
    """Checks id the index for the backfolding is in the middle of the peak

    Parameters
    ----------
    contrast : np.ndarray
        contrast array of the CASR calibration
    """
    backfold_id = find_backfolding_index(contrast)
    print(f"The backfolding index is {backfold_id}")
    for i in range(len(contrast)):
        plt.plot(contrast[i])
        plt.plot(backfold_id, contrast[i,backfold_id], 'ro')


#-----------Sensitivity calculation functions-----------------

def calc_measurement_time(yaml_config: dict) -> float:
    pulse_sequence_duration = yaml_config["duration_puseseq_cycle"] * 1e-6 # in seconds
    n_meas = yaml_config["sensor"]["config"]["number_measurements"]
    measurement_time = pulse_sequence_duration * n_meas
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


def calc_sensitivity(yaml_config: dict, data: np.ndarray, f0:float = 500, **kwargs)-> tuple[float, float, float]:
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
    prominence = kwargs.get("prominence", 0.005)
    rel_pad = kwargs.get("rel_pad", 10)
    mask_index = kwargs.get("mask_index", 20)
    window_hz = kwargs.get("window_hz", None)
    width_hz = kwargs.get("width_hz", 1)
    window_bins = kwargs.get("window_bins", 5)

    

    frequencies = calc_fourier_frequencies(yaml_config)[mask_index:]
    fft_spectrum = calc_fourier_transform(data)[mask_index:]
    idx, freq, amp = find_peak_near(frequencies, fft_spectrum, f0, window_hz=window_hz, window_bins=window_bins)
    measurement_time = calc_measurement_time(yaml_config)
    # rescale the amplitude
    fft_spectrum_normalized = fft_spectrum/amp

    noise_mask, peak_info = noise_only_mask(frequencies, fft_spectrum, prominence=prominence, rel_pad=rel_pad, width_hz=width_hz)
    std = np.std(fft_spectrum_normalized[noise_mask])
    snr = 1/std
    sensitivity = 10e-9 /snr * np.sqrt(measurement_time) # in T/√Hz

    return sensitivity, snr, std