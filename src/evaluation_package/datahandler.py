# Re-run with a loader that auto-detects ruamel.yaml or falls back to PyYAML.

from __future__ import annotations
# datahandler.py
from evaluation_package.filetools import load_yaml
import evaluation_package.casr as casr
import evaluation_package.evaluation as ev
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def get_by_dotted_path(d: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def collect_manifest(
    results_dir: Union[str, Path],
    glob_pattern: str = "CASR_sensitivity_*.yaml",
    data_ext: str = ".npy",
    param_keys: Optional[Sequence[str]] = None,
    strict_params: bool = False,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows = []
    for ypath in sorted(results_dir.glob(glob_pattern)):
        ycfg = load_yaml(ypath)
        stem = (ycfg.get("filename") if isinstance(ycfg, dict) else None) or ypath.stem
        dpath = (results_dir / stem).with_suffix(data_ext)

        row = {
            "run_id": stem,
            "yaml_path": str(ypath),
            "data_path": str(dpath),
            "yaml_exists": ypath.exists(),
            "data_exists": dpath.exists(),
            "timestamp": (ycfg.get("timestamp") if isinstance(ycfg, dict) else None) or None,
        }
        if param_keys and isinstance(ycfg, dict):
            for k in param_keys:
                val = get_by_dotted_path(ycfg, k, default=None)
                if val is None and strict_params:
                    raise KeyError(f"Missing required key '{k}' in {ypath.name}")
                row[k] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df["status"] = np.where(df["data_exists"], "ok", "missing_data")
    return df

@dataclass
class SweepSet:
    manifest: pd.DataFrame

    @classmethod
    def from_dir(
        cls,
        results_dir: Union[str, Path],
        glob_pattern: str,
        param_keys: Sequence[str],
        data_ext: str = ".npy",
        strict_params: bool = False,
        sort_by: Optional[Union[str, List[str]]] = None,
    ) -> "SweepSet":
        df = collect_manifest(
            results_dir=results_dir,
            glob_pattern=glob_pattern,
            data_ext=data_ext,
            param_keys=param_keys,
            strict_params=strict_params,
        )
        df = df[df["data_exists"]].copy()
        if sort_by is not None:
            df = df.sort_values(sort_by).reset_index(drop=True)
        return cls(df)

    def params(self, dotted_key: str) -> np.ndarray:
        if dotted_key not in self.manifest.columns:
            raise KeyError(f"Parameter '{dotted_key}' not found in manifest columns.")
        return self.manifest[dotted_key].to_numpy()

    def run_ids(self) -> List[str]:
        return self.manifest["run_id"].tolist()

    def __len__(self) -> int:
        return len(self.manifest)

    def select(self, **filters) -> "SweepSet":
        df = self.manifest
        for k, v in filters.items():
            if k not in df.columns:
                raise KeyError(f"Filter key '{k}' not found in manifest.")
            if isinstance(v, (list, tuple, set, np.ndarray)):
                df = df[df[k].isin(list(v))]
            else:
                df = df[df[k] == v]
        return SweepSet(df.reset_index(drop=True))

    def validate(self, require_same_length: bool = True) -> Dict[str, Any]:
        issues = {"missing": [], "lengths": []}
        lengths = []
        for _, row in self.manifest.iterrows():
            dpath = Path(row["data_path"])
            if not dpath.exists():
                issues["missing"].append(row["run_id"])
                continue
            try:
                arr = np.load(dpath, mmap_mode="r")
                lengths.append(int(arr.shape[-1]))
            except Exception as e:
                issues.setdefault("load_errors", []).append((row["run_id"], str(e)))
        if require_same_length and lengths:
            if len(set(lengths)) != 1:
                issues["lengths"] = lengths
        return issues

    def stack(
        self,
        mmap_mode: Optional[str] = None,
        axis: int = -1,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        arrays = []
        for _, row in self.manifest.iterrows():
            arr = np.load(row["data_path"], mmap_mode=mmap_mode)
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            arrays.append(arr)
        base_shape = arrays[0].shape
        for i, a in enumerate(arrays[1:], start=1):
            if a.shape != base_shape:
                raise ValueError(f"Shape mismatch at index {i}: {a.shape} != {base_shape}")
        stacked = np.stack(arrays, axis=0)
        return stacked

    def to_xarray(self, time_key: str = "pulse_sequence.sampling_rate", start_time: float = 0.0):
        try:
            import xarray as xr  # type: ignore
        except Exception:
            return None
        sr_col = time_key
        if sr_col not in self.manifest.columns:
            srs = []
            for yp in self.manifest["yaml_path"]:
                ycfg = load_yaml(yp)
                srs.append(get_by_dotted_path(ycfg, time_key, default=None))
            self.manifest[sr_col] = srs
        data = self.stack(mmap_mode=None)
        n_samples = data.shape[-1]
        sr = float(self.manifest[sr_col].iloc[0]) if sr_col in self.manifest.columns else None
        if sr is not None and sr > 0:
            t = start_time + np.arange(n_samples) / sr
        else:
            t = np.arange(n_samples)
        coords = {"run": np.arange(len(self)), "time": t}
        for col in self.manifest.columns:
            if col in ("run_id", "yaml_path", "data_path", "yaml_exists", "data_exists", "status", "timestamp"):
                continue
            vals = self.manifest[col].to_numpy()
            if len(vals) == len(self):
                coords[col] = ("run", vals)
        da = xr.DataArray(data, dims=("run", "time"), coords=coords, name="signal")
        return da
    


yaml_units = {
    "pulse_sequence.mixing_frequency": "MHz",
    "static_devices.mw_source.config.frequency": "Frequency/GHz",
    "pulse_sequence.N": "N",
    "pulse_sequence.pi": "Pi/ns",
    "pulse_sequence.sampling_rate": "Sampling rate/(GS/s)",
}

yaml_units_scaling_factor = {
    "pulse_sequence.mixing_frequency": 1e-6,
    "static_devices.mw_source.config.frequency": 1e-9,
    "pulse_sequence.N": 1,
    "pulse_sequence.pi": 1,
    "pulse_sequence.sampling_rate": 1e9,
}


class SweepDataHandler:
    def __init__(self, 
                 result_dir: str, 
                 yaml_key: str,
                 glob_pattern: str = "CASR_sensitivity_*.yaml",
                 param_keys: List[str] = None):
        """Initialize SweepDataHandler with directory and yaml key."""
        self.result_dir = Path(result_dir)
        self.yaml_key = yaml_key
        
        if param_keys is None:
            self.param_keys = [
                "pulse_sequence.mixing_frequency",
                "static_devices.mw_source.config.frequency",
                "pulse_sequence.N",
                "pulse_sequence.pi",
                "pulse_sequence.sampling_rate",
            ]
        else:
            self.param_keys = param_keys
            
        # Initialize data containers
        self.sweepset = None
        self.data_array = None
        self.configs = None
        self.yaml_key_params = None
        
        # Load data
        self._generate_data_handler(glob_pattern)
        self._extract_data()
        
    def _generate_data_handler(self, glob_pattern: str) -> None:
        """Generate SweepSet data handler."""
        self.sweepset = SweepSet.from_dir(
            results_dir=self.result_dir,
            glob_pattern=glob_pattern,
            param_keys=self.param_keys,
            data_ext=".npy",
            strict_params=False,
            sort_by=[self.yaml_key],
        )
        print(self.sweepset.manifest.head(20))
    

    def _extract_data(self) -> None:
        """Extract data from sweepset and group by unique parameters."""
        # Get all parameter values
        all_params = self.sweepset.params(self.yaml_key)
        
        # Get unique parameters
        self.yaml_key_params = np.array(sorted(set(all_params)))
        
        # Group data and configs
        data_list = []
        config_list = []
        
        for param in self.yaml_key_params:
            # Find indices for this parameter
            param_indices = [i for i, p in enumerate(all_params) if p == param]
            
            # Group data and configs for this parameter
            param_data = np.stack([self.sweepset.stack()[i] for i in param_indices])
            param_configs = [load_yaml(self.sweepset.manifest["yaml_path"].iloc[i]) 
                           for i in param_indices]
            
            data_list.append(param_data)
            config_list.append(param_configs)
        
        # Store as numpy arrays
        self.data_array = np.array(data_list)  # Shape: (n_params, n_reps, ...)
        self.configs = config_list  # List[List[dict]]
        self.sweep_number = len(self.yaml_key_params)

    def get_repetition_data(self, param_idx: int, rep_idx: int) -> np.ndarray:
        """Get data for specific parameter and repetition."""
        return self.data_array[param_idx][rep_idx]
      
    def get_parameter_data(self, param_idx: int) -> np.ndarray:
        """Get all repetitions for a parameter."""
        return self.data_array[param_idx]

    @property
    def parameters(self) -> List[Any]:
        """Get parameters for yaml key."""
        return self.yaml_key_params
    
    @property
    def data(self) -> np.ndarray:
        """Get data array."""
        return self.data_array
    

    def plot_fourier_spectrum(self, min_idx: int = 0, max_idx: int = -1, mask_index = 30) -> None:
        for i in np.arange(min_idx, max_idx):
            fourier_freq = casr.calc_fourier_frequencies(self.configs[i])[mask_index:]
            fourier_spectrum = casr.calc_fourier_transform(self.data_array[i])[mask_index:]
            plt.plot(fourier_freq, fourier_spectrum, label=yaml_units[self.yaml_key]+f"={self.yaml_key_params[i] * yaml_units_scaling_factor[self.yaml_key]:.4f}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fourier Amplitude (a.u.)")
        plt.legend()

    def calc_sens_snr_std(self):
        sens_list = []
        snr_list = []
        std_list = []
        for i in range(self.sweep_number):
            sens, snr, std = casr.calc_sensitivity(self.configs[i], self.data_array[i],f0=500)
            sens_list.append(sens)
            snr_list.append(snr)
            std_list.append(std)
        return np.array(sens_list), np.array(snr_list), np.array(std_list)
        

    def calc_voltage_levels(self):
        ref_levels = []
        mess_levels = []
        for i in range(self.sweep_number):
            ref, mess = ev.calc_ref_mess_voltage(self.data_array[i])
            ref_levels.append(ref)
            mess_levels.append(mess)
        return np.array(ref_levels), np.array(mess_levels)

    def plot_all_metrics(self):
        senss, snrs, stds = self.calc_sens_snr_std()
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot(self.yaml_key_params * yaml_units_scaling_factor[self.yaml_key], senss*1e12, marker='o')
        ax1.set_xlabel(yaml_units[self.yaml_key])
        ax1.set_ylabel("Sensitivity (pT/Hz^0.5)")
       
        ax2.plot(self.yaml_key_params * yaml_units_scaling_factor[self.yaml_key], snrs, marker='o', color='orange')
        ax2.set_xlabel(yaml_units[self.yaml_key])
        ax2.set_ylabel("SNR")

        ax3.plot(self.yaml_key_params * yaml_units_scaling_factor[self.yaml_key], stds, marker='o', color='green')
        ax3.set_xlabel(yaml_units[self.yaml_key])
        ax3.set_ylabel("Standard Deviation (a.u.)")

        plt.show()
    def plot_metric_vs_param(self, metric: str):
        senss, snrs, stds = self.calc_sens_snr_std()
        if metric == "sensitivity":
            plt.plot(self.yaml_key_params * yaml_units_scaling_factor[self.yaml_key], senss * 1e12, marker='o')
            plt.xlabel(yaml_units[self.yaml_key])
            plt.ylabel("Sensitivity (pT/Hz^0.5)")
        elif metric == "snr":
            plt.plot(self.yaml_key_params * yaml_units_scaling_factor[self.yaml_key], snrs, marker='o', color='orange')
            plt.xlabel(yaml_units[self.yaml_key])
            plt.ylabel("SNR")
        elif metric == "std":
            plt.plot(self.yaml_key_params * yaml_units_scaling_factor[self.yaml_key], stds, marker='o', color='green')
            plt.xlabel(yaml_units[self.yaml_key])
            plt.ylabel("Standard Deviation (a.u.)")
        else:
            raise ValueError("Metric must be one of 'sensitivity', 'snr', or 'std'.")
        plt.show()



