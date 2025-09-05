import os
import platform
from ruamel.yaml import YAML, YAMLError
import numpy as np


yaml = YAML(typ = 'safe', pure = True)
yaml.preserve_quotes = True


from datetime import datetime

def get_data_file(path: str, experiment_type: str, extension: str, date_key: str | None = None) -> str:
    """
    Return the data filename for a given experiment and extension. If `date_key` is
    provided, return the file matching that exact timestamp; otherwise return the
    latest available file.

    Expected filename pattern: `{experiment_type}_{YYYY-MM-DD-HH-MM-SS}{extension}`
    """
    ext = extension if extension.startswith('.') else f'.{extension}'
    prefix = experiment_type if experiment_type.endswith('_') else f"{experiment_type}_"

    candidates = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(ext)]
    if not candidates:
        raise FileNotFoundError(f"No files found in '{path}' matching prefix '{prefix}' and extension '{ext}'.")

    if date_key is not None:
        target = f"{prefix}{date_key}{ext}"
        if target in candidates:
            return target
        else:
            available = ", ".join(sorted(candidates))
            raise FileNotFoundError(f"Requested file '{target}' not found. Available files: {available}")

    def parse_ts(fname: str) -> datetime:
        ts = fname[len(prefix):-len(ext)]
        try:
            return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S")
        except ValueError as e:
            raise ValueError(
                f"Filename '{fname}' does not match expected timestamp format YYYY-MM-DD-HH-MM-SS."
            ) from e

    return max(candidates, key=parse_ts)

def get_datafolder_home()-> str:
    """
    Determines the root directory for data storage based on the operating system.

    Returns:
        str: The root directory path for data storage.

    Raises:
        OSError: If the operating system is not Windows or macOS.
    """
    if platform.system() == "Windows":
        directory = r"G:\Bucherlab\Sensitivity_Optimization"
    elif platform.system() == "Darwin":  # macOS
        directory = "/Volumes/001/Bucherlab/Sensitivity_Optimization"
    else:
        raise OSError("Unsupported operating system")
    return directory

def load_experiment_data(experiment_type: str, subfolder_list: list[str], date_key: str | None = None, **kwargs) -> tuple:
    """
    Loads experiment data (YAML config and NumPy array) from a specified directory.

    Filenames are expected to follow the convention:
        `{experiment_type}_{YYYY-MM-DD-HH-MM-SS}.yaml` and `.npy`
    """
    directory = os.path.join(get_datafolder_home(), *subfolder_list)

    yaml_file = get_data_file(directory, experiment_type, ".yaml", date_key)
    data_file = get_data_file(directory, experiment_type, ".npy", date_key)

    if kwargs.get("print", False):
        print("The data of the following experiment is loaded:", yaml_file)

    with open(os.path.join(directory, yaml_file), 'r') as file:
        try:
            yaml_data = yaml.load(file)
        except YAMLError as exc:
            print(f"Error in YAML file: {exc}")
            raise

    data = np.load(os.path.join(directory, data_file))
    return yaml_data, data
    