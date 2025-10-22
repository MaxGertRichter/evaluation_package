# Re-run with a loader that auto-detects ruamel.yaml or falls back to PyYAML.

from __future__ import annotations
# datahandler.py
from evaluation_package.filetools import load_yaml
import evaluation_package.casr as casr
import evaluation_package.evaluation as ev
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable
import matplotlib.pyplot as plt
import inspect
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

class SweepSet:
    def __init__(self, data_dir: str | Path, sweep_config: Dict[str, Any]):
        """
        Initialize SweepSet with a sweep configuration.
        
        Args:
            data_dir: Directory containing the generated data and YAML files
            sweep_config: The same sweep config used to generate the files
        """
        self.data_dir = Path(data_dir)
        self.sweep_config = sweep_config
        
        # Extract sweep parameters
        self.sweeps = sweep_config.get("sweeps", {})
        self.mode = sweep_config.get("mode", "cartesian")
        self.repetitions = sweep_config.get("repetitions", 1)
        
        # Collect manifest
        self.manifest = collect_manifest(self.data_dir)
        
        # Extract timestamp from filenames and add to manifest
        self.manifest['timestamp'] = self.manifest['yaml_path'].apply(
            lambda path: self._extract_timestamp(Path(path).stem) if pd.notna(path) else None
        )
        
        # Extract parameter values from YAML files
        self.param_names = list(self.sweeps.keys())
        for param_name in self.param_names:
            self.manifest[param_name] = self.manifest['yaml_path'].apply(
                lambda path: self._extract_param(path, param_name) if pd.notna(path) else None
            )
        
        # Generate expected sweep structure
        self._generate_sweep_structure()
        
        # Load and organize data
        self._data = None
        self._yaml_configs = None
    
    def _extract_timestamp(self, filename: str) -> str:
        """Extract timestamp from filename."""
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else ""
    
    def _extract_param(self, yaml_path, param_name):
        """Extract parameter value from YAML file."""
        try:
            config = load_yaml(yaml_path)
            return get_by_dotted_path(config, param_name)
        except:
            return None
    
    def _generate_sweep_structure(self):
        """Generate the expected parameter combinations based on sweep_config."""
        param_values = [self.sweeps[name] for name in self.param_names]
        
        if self.mode == "cartesian":
            from itertools import product
            self.combinations = list(product(*param_values))
            self.shape = tuple(len(v) for v in param_values) + (self.repetitions,)
        elif self.mode == "zip":
            self.combinations = list(zip(*param_values))
            self.shape = (len(self.combinations), self.repetitions)
        
    def load_data(self) -> np.ndarray:
        """
        Load all data files and organize them according to sweep structure.
        
        Returns:
            Array with shape matching the sweep structure plus data dimensions.
            For cartesian: (dim1, dim2, ..., dimN, repetitions, *data_shape)
            For zip: (n_combinations, repetitions, *data_shape)
        """
        if self._data is not None:
            return self._data
        
        # Find first data file to get shape
        first_data = None
        for _, row in self.manifest.iterrows():
            if row['data_exists']:
                first_data = np.load(row['data_path'])
                break
                
        if first_data is None:
            raise ValueError("No data files found")
            
        data_shape = first_data.shape
        full_shape = self.shape + data_shape
        
        # Initialize arrays
        self._data = np.full(full_shape, np.nan, dtype=first_data.dtype)
        self._yaml_configs = np.empty(self.shape, dtype=object)
        
        # Group by parameter combinations and sort by timestamp
        if self.param_names:
            grouped = self.manifest.groupby(self.param_names)
            
            # Process each parameter combination
            for param_values, group in grouped:
                # Make sure param_values is a tuple
                if not isinstance(param_values, tuple):
                    param_values = (param_values,)
                
                # Get index in the array
                if self.mode == "cartesian":
                    try:
                        idx = tuple(self.sweeps[name].index(val) 
                                    for name, val in zip(self.param_names, param_values))
                    except ValueError:
                        # Parameter value not in expected values
                        continue
                else:  # zip mode
                    try:
                        idx = tuple()
                        for i, combo in enumerate(self.combinations):
                            if combo == param_values:
                                idx = (i,)
                                break
                        if not idx:
                            continue
                    except:
                        continue
                
                # Sort by timestamp and assign to repetition slots
                sorted_group = group.sort_values('timestamp')
                for rep_idx, (_, row) in enumerate(sorted_group.iterrows()):
                    if rep_idx >= self.repetitions:
                        break
                        
                    if row['data_exists']:
                        full_idx = idx + (rep_idx,)
                        self._data[full_idx] = np.load(row['data_path'])
                        self._yaml_configs[full_idx] = row['yaml_path']
        
        return self._data
    
    def get_sweep_keys(self) -> Dict[str, np.ndarray]:
        """
        Get the sweep parameter values organized by dimension.
        
        Returns:
            Dictionary mapping parameter names to their values along each axis.
        """
        sweep_keys = {}
        
        if self.mode == "cartesian":
            for param_name in self.param_names:
                sweep_keys[param_name] = np.array(self.sweeps[param_name])
        else:  # zip mode
            for param_name in self.param_names:
                sweep_keys[param_name] = np.array(self.sweeps[param_name])
        
        return sweep_keys
    
    def get_yaml_path(self, *indices) -> str:
        """
        Get the YAML config path for a specific set of indices.
        
        Args:
            *indices: Indices matching the data array structure
            
        Returns:
            Path to the YAML config file
        """
        if self._yaml_configs is None:
            self.load_data()
        
        return self._yaml_configs[indices]
    
    def get_yaml_config(self, *indices) -> str:
        """
        Get the YAML config path for a specific set of indices.
        
        Args:
            *indices: Indices matching the data array structure
            
        Returns:
            Path to the YAML config file
        """
        yaml_path = self.get_yaml_path(*indices)
        return load_yaml(yaml_path)
    
    def get_parameter_grid(self) -> np.ndarray:
        """
        Generate a structured array containing parameter values at each index position.
        
        Returns:
            Array with shape (*sweep_shape, num_params) containing parameter values.
            The last dimension corresponds to parameters in the order of self.param_names.
            
        Example:
            For cartesian product with N=[14,16] and freq=[2.1e9,2.2e9] with 2 repetitions:
            The result will have shape (2, 2, 2, 2) where:
            result[0,0,0] = [14, 2.1e9]
            result[0,0,1] = [14, 2.1e9]
            result[0,1,0] = [14, 2.2e9]
            ...and so on
        """
        # Get number of parameters
        n_params = len(self.param_names)
        
        # Create output array with an extra dimension for parameters
        # Try to determine if we can use a specific dtype
        try:
            # Check if all parameter values are numeric
            all_values = []
            for values in self.sweeps.values():
                all_values.extend(values)
            
            # Try float64 as the common dtype
            sample = np.array(all_values, dtype=np.float64)
            dtype = np.float64
        except:
            # Fall back to object dtype if parameters are mixed
            dtype = object
        
        # Create output array
        param_grid = np.empty(self.shape + (n_params,), dtype=dtype)
        
        # Fill the grid based on mode
        if self.mode == "cartesian":
            # For each position in the grid
            for idx in np.ndindex(self.shape[:-1]):  # Excluding repetition dimension
                # Create array of parameter values
                param_values = np.array([
                    self.sweeps[param_name][idx[dim_idx]]
                    for dim_idx, param_name in enumerate(self.param_names)
                ], dtype=dtype)
                    
                # Assign to all repetitions
                for rep in range(self.repetitions):
                    full_idx = idx + (rep,)
                    param_grid[full_idx] = param_values
                    
        elif self.mode == "zip":
            # For each combination
            for combo_idx, combo in enumerate(self.combinations):
                # Convert combo to numpy array
                param_values = np.array(combo, dtype=dtype)
                
                # Assign to all repetitions
                for rep in range(self.repetitions):
                    param_grid[combo_idx, rep] = param_values
        
        return param_grid


    def get_yaml_configs_array(self) -> np.ndarray:
        """
        Get the full array of YAML config paths matching the data structure.
        
        Returns:
            Array of shape matching sweep structure (without data dimensions)
        """
        if self._yaml_configs is None:
            self.load_data()
        
        return self._yaml_configs
    

    def query(self, keys: List[str], values: List[Any]) -> np.ndarray:
        """
        Query the data array for entries matching specific parameter values.
        
        Args:
            keys: List of parameter names to match (e.g., ["pulse_sequence.N", "pulse_sequence.awg_frequency"])
            values: List of corresponding values to filter by (e.g., [16, 2.1909e9])
        
        Returns:
            Numpy array with all data entries matching the specified parameter values
        """
        # Ensure data is loaded
        if self._data is None:
            self.load_data()
            
        # Validate keys
        for key in keys:
            if key not in self.param_names:
                raise ValueError(f"Parameter '{key}' not found in sweep configuration")
        
        # Get parameter indices in param_names list
        param_indices = [self.param_names.index(key) for key in keys]
        
        # Get parameter grid
        param_grid = self.get_parameter_grid()
        
        # Create mask for matching entries
        mask = np.ones(self.shape, dtype=bool)
        
        # Apply filters for each key-value pair
        for i, (param_idx, value) in enumerate(zip(param_indices, values)):
            # Check each position if its parameter value matches
            param_mask = param_grid[..., param_idx] == value
            mask = mask & param_mask
        
        # Get indices of matches
        match_indices = np.where(mask)
        
        # Extract data at matching indices
        if len(match_indices[0]) == 0:
            return np.array([])  # No matches
        
        # Gather matching data
        result = np.array([self._data[idx] for idx in zip(*match_indices)])
        
        # Reshape result based on number of dimensions in data
        data_shape = self._data.shape[len(self.shape):]  # Get shape of data part
        result_shape = (len(match_indices[0]),) + data_shape
        
        return result.reshape(result_shape)


    def process_data(self, func: Callable, **kwargs) -> np.ndarray:
        """
        Apply a processing function to each data point in the sweep.
        
        Args:
            func: Function to process data. Supported signatures:
                - func(data, **kwargs)
                - func(data, yaml_config, **kwargs)
                - func(data, params, **kwargs)
                - func(data, yaml_config, params, **kwargs)
                - func(yaml_config, **kwargs)  # For config-only processing
            **kwargs: Additional keyword arguments passed to func
        
        Returns:
            Array with same shape as data array except the data dimensions are 
            replaced with the shape of the function output
        """
        
        # Ensure data is loaded
        if self._data is None:
            self.load_data()
        
        # Inspect function signature to determine needed arguments
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        # Check what the function needs
        needs_data = 'data' in param_names
        needs_yaml = 'yaml_config' in param_names
        needs_params = 'params' in param_names
        
        # Get parameter grid if needed
        param_grid = self.get_parameter_grid() if needs_params else None
        
        # Process first non-nan item to determine output shape
        sample_idx = None
        for idx in np.ndindex(self.shape):
            if not np.isnan(self._data[idx]).all():
                sample_idx = idx
                break
                
        if sample_idx is None:
            raise ValueError("All data points are NaN")
        
        # Build argument dictionary for sample execution
        sample_args = {}
        if needs_data:
            sample_args['data'] = self._data[sample_idx]
        if needs_yaml:
            sample_args['yaml_config'] = self.get_yaml_config(*sample_idx)
        if needs_params:
            sample_args['params'] = param_grid[sample_idx] if param_grid is not None else None
        sample_args.update(kwargs)
        
        # Execute function on sample to determine output shape
        sample_output = func(**sample_args)
        
        # Determine output shape
        if isinstance(sample_output, (int, float, bool, str)):
            output_shape = self.shape + (1,)
            output_dtype = type(sample_output)
        elif isinstance(sample_output, np.ndarray):
            output_shape = self.shape + sample_output.shape
            output_dtype = sample_output.dtype
        else:
            try:
                sample_output = np.array(sample_output)
                output_shape = self.shape + sample_output.shape
                output_dtype = sample_output.dtype
            except:
                output_shape = self.shape + (1,)
                output_dtype = object
        
        # Create output array
        result = np.empty(output_shape, dtype=output_dtype)
        
        # Process all data
        for idx in np.ndindex(self.shape):
            # Build argument dictionary
            args = {}
            if needs_data:
                args['data'] = self._data[idx]
            if needs_yaml:
                args['yaml_config'] = self.get_yaml_config(*idx)
            if needs_params:
                args['params'] = param_grid[idx] if param_grid is not None else None
            args.update(kwargs)
            
            # Skip if needed data is NaN
            if needs_data and isinstance(args['data'], np.ndarray) and np.isnan(args['data']).all():
                if isinstance(sample_output, np.ndarray):
                    result[idx] = np.full(sample_output.shape, np.nan)
                else:
                    result[idx] = np.nan
                continue
            
            # Apply function
            output = func(**args)
            
            # Store result
            if isinstance(output, (int, float, bool, str)) and output_shape[-1] == 1:
                result[idx] = output
            else:
                result[idx] = np.array(output)
        
        return result

    @property
    def data(self) -> np.ndarray:
        """Access the loaded data array."""
        if self._data is None:
            self.load_data()
        return self._data


"""@dataclass
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
        return da"""
    


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



