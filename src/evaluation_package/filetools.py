import os
import platform
from ruamel.yaml import YAML, YAMLError
import numpy as np


yaml = YAML(typ = 'safe', pure = True)
yaml.preserve_quotes = True


# Import data with given name pattern and latest date from directory      
def get_latest_data_file(path, experiment_type, ending):
    file_list = os.listdir(path) #creates list of all files in directory
    date_list = []
    for i in file_list:
        if experiment_type in i: 
            if ending in i:
                date = i.replace(experiment_type, '').replace(ending, '')
                date_list.append(date) # creates list of all dates without string
    recent_date = experiment_type+max(date_list)+ending # creates the recent date, by searching the max of date_list
    return(recent_date)

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

def load_experiment_data(experiment_type, subfolder_list, **kwargs) -> tuple:
    """
    Loads experiment data from a specified directory based on the experiment type and subfolder structure.

    This function retrieves the latest YAML and NumPy data files from the specified directory,
    parses the YAML file for metadata, and loads the NumPy file for experiment data.

    Args:
        experiment_type (str): The type of experiment, used to identify the relevant files.
        subfolder_list (list of str): A list of subfolder names specifying the path to the data directory.

    Returns:
        tuple:
            - yaml_data (dict): Parsed data from the YAML file.
            - data (numpy.ndarray): Loaded data from the NumPy file.

    Raises:
        YAMLError: If there is an error while parsing the YAML file.
        FileNotFoundError: If the required YAML or NumPy file is not found in the directory.
    """
    directory = os.path.join(get_datafolder_home(), *subfolder_list)
    yaml_file = get_latest_data_file(directory, experiment_type, 'yaml')
    data_file = get_latest_data_file(directory, experiment_type, 'npy')
    for key, value in kwargs.items():
        if key == "print" and value:
            print("The data of the following experiment is loaded:" , yaml_file)
    with open(os.path.join(directory, yaml_file), 'r') as file:
        try:
            yaml_data = yaml.load(file)
        except YAMLError as exc:
            print(f"Error in YAML file: {exc}")
            return None
    data = np.load(os.path.join(directory, data_file))
    return yaml_data, data
    