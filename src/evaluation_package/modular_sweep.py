from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional
from itertools import product
from pathlib import Path
from datetime import datetime
from evaluation_package.filetools import load_yaml, save_yaml
from evaluation_package.param_sweep import set_by_dotted_path, format_value_for_filename

def _expand_dotted_path(dotted: str, value: Any) -> Dict[str, Any]:
    """Expands a dotted path a.b.c into {a: {b: {c: value}}}."""
    parts = dotted.split(".")
    # Iterate backwards to build the dict from inside out
    curr = value
    for p in reversed(parts):
        curr = {p: curr}
    return curr

def _merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges d2 into d1."""
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            _merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1

class SweepStrategy(ABC):
    """Abstract base class for sweep strategies."""
    
    @abstractmethod
    def generate(self) -> Iterator[Dict[str, Any]]:
        """Yields dictionaries of parameter updates."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the sweep (type and parameters)."""
        pass

class LinearSweep(SweepStrategy):
    """
    Sweeps a single parameter over a list of values.
    
    Example:
        LinearSweep("pulse_sequence.N", [1, 2, 3])
    """
    def __init__(self, parameter_name: str, values: List[Any]):
        self.parameter_name = parameter_name
        self.values = values

    def generate(self) -> Iterator[Dict[str, Any]]:
        for v in self.values:
            yield {self.parameter_name: v}

    def get_metadata(self) -> Dict[str, Any]:
        meta = {"type": "linear"}
        param_meta = _expand_dotted_path(self.parameter_name, self.values)
        return _merge_dicts(meta, param_meta)

class ZipSweep(SweepStrategy):
    """
    Sweeps multiple parameters in lock-step (zipped).
    All value lists must be of the same length.
    
    Example:
        ZipSweep({
            "param1": [1, 2],
            "param2": [10, 20]
        })
        Yields: {"param1": 1, "param2": 10}, {"param1": 2, "param2": 20}
    """
    def __init__(self, parameters: Dict[str, List[Any]]):
        self.parameters = parameters
        lengths = [len(v) for v in parameters.values()]
        if not lengths:
            raise ValueError("No parameters provided for ZipSweep")
        if len(set(lengths)) != 1:
            raise ValueError(f"ZipSweep requires equal lengths for all parameters. Got lengths: {lengths}")
        self.length = lengths[0]

    def generate(self) -> Iterator[Dict[str, Any]]:
        keys = list(self.parameters.keys())
        # zip corresponding values
        for i in range(self.length):
            yield {k: self.parameters[k][i] for k in keys}

    def get_metadata(self) -> Dict[str, Any]:
        meta = {"type": "zip"}
        for k, v in self.parameters.items():
            param_meta = _expand_dotted_path(k, v)
            _merge_dicts(meta, param_meta)
        return meta

class CartesianSweep(SweepStrategy):
    """
    Sweeps multiple parameters in all combinations (Cartesian product).
    
    Example:
        CartesianSweep({
            "param1": [1, 2],
            "param2": [10, 20]
        })
        Yields 4 combinations.
    """
    def __init__(self, parameters: Dict[str, List[Any]]):
        self.parameters = parameters

    def generate(self) -> Iterator[Dict[str, Any]]:
        keys = list(self.parameters.keys())
        values_list = [self.parameters[k] for k in keys]
        for combo in product(*values_list):
            yield dict(zip(keys, combo))

    def get_metadata(self) -> Dict[str, Any]:
        meta = {"type": "cartesian"}
        for k, v in self.parameters.items():
            param_meta = _expand_dotted_path(k, v)
            _merge_dicts(meta, param_meta)
        return meta

def generate_configs(
    base_yaml_path: Path,
    sweep_strategy: SweepStrategy,
    output_dir: Path,
    file_prefix_index: bool = True,
    start_index: int = 1
) -> List[Path]:
    """
    Generates configuration files based on the sweep strategy.
    
    Args:
        base_yaml_path: Path to the template YAML file.
        sweep_strategy: The sweep strategy (Linear, Zip, or Cartesian).
        output_dir: Directory to save generated files.
        file_prefix_index: Whether to prefix filenames with an index (01__, 02__, etc.).
        start_index: Starting index for numbering.
        
    Returns:
        List of paths to generated files.
    """
    base_yaml_path = Path(base_yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_stem = base_yaml_path.stem
    generated_files = []
    
    # Get metadata once
    sweep_metadata = sweep_strategy.get_metadata()
    
    # Generate a unique key for this batch of files
    sweep_key = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sweep_metadata["sweep_key"] = sweep_key
    
    # We consume the generator to list to know total count for padding
    configs = list(sweep_strategy.generate())
    total = len(configs)
    pad_width = max(2, len(str(start_index + total - 1)))
    
    current_index = start_index
    
    for params in configs:
        # Load fresh config
        cfg = load_yaml(base_yaml_path)
        
        # Inject metadata
        cfg["zsweep"] = sweep_metadata
        
        # Update config
        filename_parts = []
        for key, value in params.items():
            set_by_dotted_path(cfg, key, value)
            # Create filename part: last part of key - value
            short_key = key.split(".")[-1]
            val_str = format_value_for_filename(value)
            filename_parts.append(f"{short_key}-{val_str}")
            
        # Construct filename
        # Note: The order of filename parts depends on dictionary iteration order,
        # which is insertion-ordered in modern Python.
        middle = "__".join(filename_parts)
        
        if file_prefix_index:
            idx_str = str(current_index).zfill(pad_width)
            fname = f"{idx_str}__{base_stem}__{middle}.yaml"
        else:
            fname = f"{base_stem}__{middle}.yaml"
            
        out_path = output_dir / fname
        save_yaml(cfg, out_path)
        generated_files.append(out_path)
        current_index += 1
        
    return generated_files
