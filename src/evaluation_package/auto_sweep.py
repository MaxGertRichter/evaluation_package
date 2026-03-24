import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import glob
import re
import inspect
import os
from itertools import product
from .filetools import load_yaml

class AutoSweepLoader:
    def __init__(self, data_dir: Union[str, Path], yaml_pattern: str = "*.yaml", sweep_key: Optional[str] = None):
        """
        Automatically loads and organizes sweep data from a directory based on 'zsweep' metadata.
        
        If `sweep_key` is provided, only loads files matching that key.
        If `sweep_key` is NOT provided, it defaults to the `sweep_key` found in the most recently 
        modified YAML file. If the most recent file has no `sweep_key`, a ValueError is raised.

        Args:
            data_dir: Directory containing YAML configuration files and .npy data files.
            yaml_pattern: Glob pattern to identify YAML files (default: "*.yaml").
            sweep_key: Optional unique identifier for a specific sweep generation batch.
        """
        self.data_dir = Path(data_dir)
        self.yaml_pattern = yaml_pattern
        
        # 1. Find all YAMLs
        all_yaml_files = sorted(list(self.data_dir.glob(self.yaml_pattern)))
        if not all_yaml_files:
            raise FileNotFoundError(f"No YAML files found in {self.data_dir} matching {self.yaml_pattern}")

        # 2. Filter based on sweep_key
        if sweep_key:
            self.yaml_files = self._filter_yamls_by_key(all_yaml_files, sweep_key)
            if not self.yaml_files:
                raise ValueError(f"No YAML files found with sweep_key='{sweep_key}' in {self.data_dir}")
        else:
            # Automatic detection from most recent file
            # Sort by modification time (newest last)
            latest_file = max(all_yaml_files, key=os.path.getmtime)
            
            latest_meta = self._get_zsweep_metadata_from_file(latest_file)
            
            if latest_meta and 'sweep_key' in latest_meta:
                detected_key = latest_meta['sweep_key']
                print(f"Auto-detected sweep_key='{detected_key}' from most recent file '{latest_file.name}'.")
                self.yaml_files = self._filter_yamls_by_key(all_yaml_files, detected_key)
            else:
                # Strict fallback: Error if key is missing in latest file
                raise ValueError(
                    f"Most recent file '{latest_file.name}' does not contain a 'sweep_key' in 'zsweep' metadata. "
                    "Cannot automatically determine which files to load. \n"
                    "Please provide a specific `sweep_key` argument or ensure your data was generated with the updated modular sweep system."
                )

        # 3. Extract sweep metadata (from the first valid file in our filtered list)
        self.sweep_meta = self._detect_sweep_metadata()
        self.sweep_type = self.sweep_meta.get('type')
        # Handle nested parameters
        self.output_params = self._extract_params_from_meta(self.sweep_meta)
        
        # 4. Build detailed manifest (map params to files)
        self.manifest = self._build_manifest()
        
        # 5. Organize data into structure
        self.data, self.coords = self._organize_data()

    def _get_zsweep_metadata_from_file(self, yaml_path: Path) -> Optional[Dict[str, Any]]:
        """Safe helper to read zsweep metadata from a single file."""
        try:
            cfg = load_yaml(yaml_path)
            return cfg.get('zsweep')
        except Exception:
            return None

    def _filter_yamls_by_key(self, yaml_files: List[Path], key: str) -> List[Path]:
        """Returns only the files that match the given sweep_key."""
        filtered = []
        for yf in yaml_files:
            meta = self._get_zsweep_metadata_from_file(yf)
            if meta and meta.get('sweep_key') == key:
                filtered.append(yf)
        return sorted(filtered)

    def _extract_params_from_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts parameters from zsweep metadata.
        Handles both flat `{'param.name': [vals]}` and nested `{'param': {'name': [vals]}}` structures.
        Excludes 'type' key.
        """
        params = {}
        
        def _flatten(d: Dict[str, Any], prefix: str = ''):
            for k, v in d.items():
                if k == 'type' and prefix == '':
                    continue
                
                new_key = f"{prefix}.{k}" if prefix else k
                
                if isinstance(v, dict):
                    _flatten(v, new_key)
                else:
                    # Assume this is a parameter value list
                    params[new_key] = v
                    
        _flatten(meta)
        return params

    def _detect_sweep_metadata(self) -> Dict[str, Any]:
        """Reads the 'zsweep' key from the first YAML that has it."""
        for yf in self.yaml_files:
            try:
                # filetools.load_yaml takes a Path object
                cfg = load_yaml(yf)
                if 'zsweep' in cfg:
                    return cfg['zsweep']
            except Exception:
                continue
        raise ValueError("Could not find 'zsweep' metadata in any YAML file. Ensure files were generated with the new modular_sweep system.")

    def _get_param_value(self, cfg: Dict[str, Any], param_path: str) -> Any:
        """Helper to extract a dotted parameter path from a dict."""
        keys = param_path.split('.')
        val = cfg
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return None

    def _build_manifest(self) -> pd.DataFrame:
        """
        Scans all YAMLs, extracts their sweep parameter values, and identifies associated .npy files.
        Returns a DataFrame where each row is a configuration (one YAML), with columns for parameters,
        file path, and associated data files.
        """
        records = []
        param_names = list(self.output_params.keys())
        
        for yf in self.yaml_files:
            try:
                cfg = load_yaml(yf)
            except Exception as e:
                print(f"Warning: Failed to load {yf}: {e}")
                continue
            
            # Extract values for the sweep parameters from the actual file content
            row = {'yaml_path': yf}
            for p in param_names:
                row[p] = self._get_param_value(cfg, p)
            
            # Find associated .npy files
            # Robust matching using timestamp
            stem = yf.stem
            # Timestamp format: YYYY-MM-DD-HH-MM-SS (19 chars)
            # Standard name: Experiment_Timestamp
            # Chunk name: Experiment_ch-X_Timestamp
            try:
                if len(stem) > 20 and stem[-20] == '_':
                    timestamp = stem[-19:]
                    base = stem[:-20]
                    # This pattern matches both standard and chunked files
                    pattern = f"{base}*{timestamp}.npy"
                    npy_files = sorted(list(self.data_dir.glob(pattern)))
                else:
                    # Fallback
                    npy_files = sorted(list(self.data_dir.glob(f"{stem}*.npy")))
            except Exception:
                npy_files = sorted(list(self.data_dir.glob(f"{stem}*.npy")))
            
            row['npy_files'] = npy_files
            row['num_chunks'] = len(npy_files)
            
            records.append(row)
            
        df = pd.DataFrame(records)
        
        # Sort by parameters to ensure logical order
        if param_names:
            df = df.sort_values(by=param_names)
            
        return df

    def _organize_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Loads .npy files and structures them into a grid.
        Returns:
            data: The n-dimensional data array.
            coords: A dictionary mapping parameter names to their sorted unique values (the grid axes).
        """
        param_names = list(self.output_params.keys())
        
        # 1. Determine Grid Dimensions
        coords = {}
        for p in param_names:
            unique_vals = sorted(self.manifest[p].unique())
            coords[p] = np.array(unique_vals)
            
        # Determine shape based on sweep type
        if self.sweep_type == 'cartesian':
            grid_shape = tuple(len(coords[p]) for p in param_names)
        elif self.sweep_type in ['linear', 'zip']:
            grid_shape = (len(self.manifest),)
        else:
            grid_shape = (len(self.manifest),)

        # 2. Iterate and Load
        temp_grid = {} # Map global index tuple -> list of loaded chunks
        
        max_reps = 0
        
        for _, row in self.manifest.iterrows():
            # Calculate grid index
            if self.sweep_type == 'cartesian':
                idx = []
                for p in param_names:
                    val = row[p]
                    # Find index of val in coords[p]
                    p_idx = list(coords[p]).index(val)
                    idx.append(p_idx)
                grid_idx = tuple(idx)
            else:
                # Linear/Zip
                grid_idx = (self.manifest.index.get_loc(row.name),) 

            # Load data chunks
            npy_files = row['npy_files']
            loaded_chunks = []
            for nf in npy_files:
                try:
                    arr = np.load(nf)
                    loaded_chunks.append(arr)
                except Exception as e:
                    print(f"Warning: Failed to load {nf}: {e}")
            
            # Robust sort by chunk index
            try:
                def extract_chunk_idx(p):
                    m = re.search(r'_ch-(\d+)_', p.name)
                    return int(m.group(1)) if m else -1
                
                # Check if we have chunk indices
                if any('_ch-' in f.name for f in npy_files):
                    # Sort files first
                    npy_files_sorted = sorted(npy_files, key=extract_chunk_idx)
                    # Reload in correct order
                    loaded_chunks = [np.load(nf) for nf in npy_files_sorted]
            except Exception:
                pass # Keep original glob sort if extraction fails
            
            temp_grid[grid_idx] = loaded_chunks
            max_reps = max(max_reps, len(loaded_chunks))

        # 3. Consolidate
        
        # Check first data item to decide strategy
        all_chunks_flat = [c for chunks in temp_grid.values() for c in chunks]
        shapes_consistent = True
        ref_shape = None
        
        if all_chunks_flat:
            ref_shape = all_chunks_flat[0].shape
            if any(c.shape != ref_shape for c in all_chunks_flat):
                shapes_consistent = False
        
        # If we have NO data, return empty
        if not all_chunks_flat:
             return np.array([]), coords

        if shapes_consistent and ref_shape is not None:
            # Create a huge array
            # Shape: (*grid_shape, max_reps, *ref_shape)
            final_shape = grid_shape + (max_reps,) + ref_shape
            full_arr = np.zeros(final_shape, dtype=all_chunks_flat[0].dtype)
            
            for grid_idx, chunks in temp_grid.items():
                for r_idx, chunk in enumerate(chunks):
                    full_idx = grid_idx + (r_idx,)
                    full_arr[full_idx] = chunk
            
            return full_arr, coords
            
        else:
            # Fallback to Object Array
            print("Notice: Data shapes are inconsistent or jagged. Returning object array container.")
            
            obj_arr = np.empty(grid_shape, dtype=object)
            
            for grid_idx, chunks in temp_grid.items():
                try:
                    if chunks:
                        point_data = np.stack(chunks, axis=0) # (Reps, ...)
                    else:
                        point_data = np.array([])
                except ValueError:
                    point_data = np.array(chunks, dtype=object)
                
                obj_arr[grid_idx] = point_data
                
            return obj_arr, coords

    def apply(self, func: Callable, by_chunk: bool = True, **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Applies a function to each dataset in the sweep.
        
        The function `func` is called for every configuration in the sweep.
        It attempts to pass `yaml_config` and `data` as arguments if the function accepts them.
        
        Args:
            func: The function to apply. Should accept `yaml_config` (dict) and/or `data` (np.ndarray).
            by_chunk: If True, iterates over the first dimension of the data (repetitions/chunks)
                      and applies the function to each chunk individually.
                      If False (default), passes the entire data stack for the sweep point to the function.
            **kwargs: Additional keyword arguments passed to `func`.
            
        Returns:
            A tuple (result_data, coords), where result_data has the same structure as the sweep.
        """
        sig = inspect.signature(func)
        wants_config = 'yaml_config' in sig.parameters
        wants_data = 'data' in sig.parameters
        
        # Determine grid shape
        if self.sweep_type == 'cartesian':
            param_names = list(self.output_params.keys())
            grid_shape = tuple(len(self.coords[p]) for p in param_names)
        else:
            grid_shape = (len(self.manifest),)
            
        results = np.empty(grid_shape, dtype=object)
        temp_results = {}
        
        # Iterate over manifest to match data and configs
        for _, row in self.manifest.iterrows():
            # 1. Calculate Grid Index (Same logic as _organize_data)
            if self.sweep_type == 'cartesian':
                g_idx = []
                for p in self.output_params.keys():
                    val = row[p]
                    p_val_idx = list(self.coords[p]).index(val)
                    g_idx.append(p_val_idx)
                grid_idx = tuple(g_idx)
            else:
                # Linear/Zip: Just row index in sorted df
                grid_idx = (self.manifest.index.get_loc(row.name),)

            # 2. Get Data Slice
            try:
                # Retrieve the data slice for this grid index from self.data
                # We need to slice self.data at grid_idx
                # self.data shape: (*grid_shape, Reps, ...)
                # So we take self.data[grid_idx]
                current_data = self.data[grid_idx]
            except Exception:
                current_data = None
                
            # 3. Load Config
            yf = Path(row['yaml_path'])
            try:
                yaml_config = load_yaml(yf)
            except Exception as e:
                print(f"Error loading yaml {yf}: {e}")
                yaml_config = {}

            # 4. Call Function
            call_args = kwargs.copy()
            if wants_config:
                call_args['yaml_config'] = yaml_config
            
            res = None
            try:
                if by_chunk and current_data is not None:
                    # Iterate over the first dimension (Reps)
                    chunk_results = []
                    # Handle object array or numeric array
                    # current_data should be iterable over axis 0
                    for chunk in current_data:
                        if wants_data:
                            call_args['data'] = chunk
                        
                        r = func(**call_args)
                        chunk_results.append(r)
                    
                    # Stack chunk results if possible
                    try:
                        res = np.array(chunk_results)
                    except Exception:
                         res = chunk_results # Keep as list if jagged
                else:
                    # Pass the whole stack
                    if wants_data:
                        call_args['data'] = current_data
                    
                    res = func(**call_args)
            except Exception as e:
                print(f"Error applying function at {grid_idx}: {e}")
                res = None
                
            temp_results[grid_idx] = res
            
        # Fill results array
        for idx, val in temp_results.items():
            results[idx] = val
            
        # Try to convert object array to numeric if shapes match
        try:
            sample_res = results.flat[0]
            if sample_res is not None and hasattr(sample_res, 'shape'):
                ref_shape = sample_res.shape
                # Verify all elements have the same shape
                all_match = True
                for r in results.flat:
                    if r is None or not hasattr(r, 'shape') or r.shape != ref_shape:
                        all_match = False
                        break
                
                if all_match:
                    # Create numeric array
                    # Shape: (*grid_shape, *ref_shape)
                    final_shape = grid_shape + ref_shape
                    # Determine dtype
                    dt = sample_res.dtype
                    new_results = np.zeros(final_shape, dtype=dt)
                    
                    for idx, val in temp_results.items():
                        new_results[idx] = val
                    return new_results, self.coords
        except Exception:
            pass # Keep as object array
            
        return results, self.coords

    def get_data(self) -> np.ndarray:
        return self.data
