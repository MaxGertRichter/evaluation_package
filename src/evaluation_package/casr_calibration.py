import numpy as np
from scipy import constants
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from evaluation_package import utils as ut

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
    backfold_id = dict_minmax["slow_minima"][1]
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
    contrast = ut.contrast(data)
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