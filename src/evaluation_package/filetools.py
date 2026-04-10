import os
import platform
from ruamel.yaml import YAML, YAMLError
import numpy as np
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Union
import shutil
from .config import config

yaml = YAML(typ = 'safe', pure = True)
yaml.preserve_quotes = True

from datetime import datetime

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.load(f)



def save_yaml(data, path):
    buf = StringIO()
    yaml.dump(data, buf)
    text = buf.getvalue()
    with open(path, "w") as f:
        f.write(text)


def get_data_file(path: Union[Path, str], experiment_type: str, extension: str, date_key: str | None = None) -> str:
    """
    Return the data filename for a given experiment and extension. If `date_key` is
    provided, return the file matching that exact timestamp; otherwise return the
    latest available file.

    Expected filename pattern: `{experiment_type}_{YYYY-MM-DD-HH-MM-SS}{extension}`
    """
    ext = extension if extension.startswith('.') else f'.{extension}'
    prefix = experiment_type if experiment_type.endswith('_') else f"{experiment_type}_"
    
    path_obj = Path(path)
    candidates = [f.name for f in path_obj.iterdir() if f.name.startswith(prefix) and f.name.endswith(ext)]
    
    if not candidates:
        raise FileNotFoundError(f"No files found in '{path}' matching prefix '{prefix}' and extension '{ext}'.")

    if date_key is not None:
        target_suffix = f"{date_key}{ext}"
        matches = [c for c in candidates if c.endswith(target_suffix)]
        target = f"*{target_suffix}"
        if matches:
            return matches[0]
        else:
            available = ", ".join(sorted(candidates))
            raise FileNotFoundError(f"Requested file '{target}' not found. Available files: {available}")

    def parse_ts(fname: str) -> datetime:
        len_ts = len("YYYY-MM-DD-HH-MM-SS")
        ts = fname[-len_ts-len(ext):-len(ext)]
        try:
            return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S")
        except ValueError as e:
            raise ValueError(
                f"Filename '{fname}' does not match expected timestamp format YYYY-MM-DD-HH-MM-SS."
            ) from e

    return max(candidates, key=parse_ts)

def get_datafolder_home()-> str:
    """
    Determines the root directory for data storage based on the configuration file.

    Returns:
        str: The root directory path for data storage.
    """
    return str(config.data_folder_home)

def load_experiment_data(experiment_type: str, subfolders: Union[str, list, tuple], date_key: str | None = None, **kwargs) -> tuple:
    """
    Loads experiment data (YAML config and NumPy array) from a specified directory.
    `subfolders` can be a single relative path string like "01_Datafolder/01_Random_runs" 
    or a list/tuple of parts like ["01_Datafolder", "01_Random_runs"].

    Filenames are expected to follow the convention:
        `{experiment_type}_{YYYY-MM-DD-HH-MM-SS}.yaml` and `.npy`
    """
    if isinstance(subfolders, (list, tuple)):
        directory = Path(get_datafolder_home()).joinpath(*subfolders)
    else:
        # pathlib handles forward/backward slash conversions implicitly based on OS
        directory = Path(get_datafolder_home()) / subfolders

    yaml_file = get_data_file(directory, experiment_type, ".yaml", date_key)
    data_file = get_data_file(directory, experiment_type, ".npy", date_key)

    if kwargs.get("print", False):
        print("The data of the following experiment is loaded:", yaml_file)

    with open(directory / yaml_file, 'r') as file:
        try:
            yaml_data = yaml.load(file)
        except YAMLError as exc:
            print(f"Error in YAML file: {exc}")
            raise
    data = []
    if "save_in_chunks" in yaml_data["data"]:
        for i in np.arange(yaml_data["averages"]//yaml_data["data"]["save_in_chunks"]):
            chunk_file = data_file.replace(experiment_type+"_ch-0", experiment_type+f"_ch-{i}")
            print(f"Loading chunk file: {chunk_file}")
            chunk_data = np.load(directory / chunk_file)
            data.append(chunk_data)
            
    
    data.append(np.load(directory / data_file))
    return yaml_data, data
    
# Sync the yaml files in a directory with a master yaml file
def sync_yaml_parameters(master_path: Path, target_dir: Path) -> None:
    """Sync parameters while preserving yaml formatting."""
    
    # Initialize YAML with roundtrip mode
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    # Load master yaml
    try:
        with open(master_path) as f:
            master_config = yaml.load(f)
    except YAMLError as e:
        raise ValueError(f"Error loading master yaml: {e}")

    def _update_nested_dict(target: Dict, source: Dict) -> None:
        """Update nested dictionary preserving structure."""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _update_nested_dict(target[key], value)
                else:
                    target[key] = value

    # Find yaml files
    yaml_files = list(target_dir.glob("*.yaml"))
    yaml_files = [f for f in yaml_files if f != master_path]
    
    updated_files = []
    
    # Update files preserving format
    for yaml_path in yaml_files:
        try:
            with open(yaml_path) as f:
                target_config = yaml.load(f)
            _update_nested_dict(target_config, master_config)
            with open(yaml_path, 'w') as f:
                yaml.dump(target_config, f)
            updated_files.append(yaml_path)
        except YAMLError as e:
            print(f"Error processing {yaml_path}: {e}")
            continue
            
    return None


def move_files(src_dir: str, dst_dir: str, overwrite: bool = False) -> None:
    """Move all files from source to destination directory."""
    
    # Convert to Path objects
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Move each file
    for item in src_path.glob('*'):
        try:
            dst_file = dst_path / item.name
            if dst_file.exists() and not overwrite:
                print(f"Skipping {item.name} - already exists")
                continue
                
            shutil.move(str(item), str(dst_file))
            print(f"Moved {item.name}")
            
        except Exception as e:
            print(f"Error moving {item.name}: {e}")