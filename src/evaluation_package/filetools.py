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
    import re
    ext = extension if extension.startswith('.') else f'.{extension}'
    prefix = experiment_type

    # build regex: starts with experiment_type, ends with _date_key.ext
    if date_key is not None:
        pattern = re.compile(rf"^{re.escape(prefix)}.*_{re.escape(date_key)}{re.escape(ext)}$")
    else:
        # match any file that starts with experiment_type and ends with extension
        pattern = re.compile(rf"^{re.escape(prefix)}.*{re.escape(ext)}$")

    path_obj = Path(path)
    candidates = [f.name for f in path_obj.iterdir() if pattern.match(f.name)]
    
    if not candidates:
        raise FileNotFoundError(f"No files found in '{path}' matching prefix '{prefix}' and extension '{ext}'.")

    date_pattern = re.compile(rf"_(\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{2}}){re.escape(ext)}$")

    def extract_date(filename: str) -> datetime:
        match = date_pattern.search(filename)
        if not match:
            return datetime.min
        return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")

    candidates.sort(key=lambda name: (extract_date(name), name))

    return candidates[-1]  # exact match found

DATA_FOLDER_HOME = str(config.data_folder_home)

def get_datafolder_home()-> str:
    """
    Determines the root directory for data storage based on the configuration file.

    Returns:
        str: The root directory path for data storage.
    """
    return DATA_FOLDER_HOME

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
    import re
    data = []
    if "save_in_chunks" in yaml_data["data"]:
        # Use regex to find and replace the chunk index marker (_ch-N_)
        chunk_pattern = re.compile(r"_ch-\d+_")
        num_chunks = int(yaml_data["averages"] // yaml_data["data"]["save_in_chunks"])
        
        for i in range(num_chunks):
            # Transform whatever chunk index was found (e.g. _ch-9_) into the current loop index
            chunk_file = chunk_pattern.sub(f"_ch-{i}_", data_file)
            
            if kwargs.get("print", False):
                print(f"Loading chunk file: {chunk_file}")
            
            chunk_data = np.load(directory / chunk_file)
            data.append(chunk_data)
    else:
        # Standard non-chunked experiment
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