# Evaluation Package

A Python library for the evaluation and analysis of Nitrogen-Vacancy (NV) center quantum sensing experiments. This package provides data processing pipelines for standard measurement protocols, from raw data ingestion to spectral analysis and parameter extraction.

## Features

- **ESR Analysis (`esr.py`)**: Lorentzian fitting, resonance frequency extraction, and contrast calculation for Electron Spin Resonance spectra.
- **Rabi Oscillations (`rabi.py`)**: Extraction of $\pi$-pulse durations, Rabi frequencies, and signal contrast sequences.
- **CASR Protocols (`casr.py` & `casr_calibration.py`)**: Fast Fourier Transform (FFT) methods for Continuous AC Sensing and Ranging. Includes functions for sensitivity calculation, noise floor estimation, and RF amplitude calibration.
- **Data Management (`filetools.py`)**: Automated loading and formatting of multidimensional NumPy arrays coupled with YAML configuration definitions.
- **Cross-Platform Compatibility**: Path resolution natively supports experimental data structures across Windows, macOS, and Linux.

## Installation

Clone the repository and install the package via pip in editable mode to ensure changes propagate to operational environments:

```bash
git clone https://github.com/MaxGertRichter/evaluation_package.git
cd evaluation_package
pip install -e .
```

## Quick Start
Example templates and datasets are provided to test functionality. 

1. Navigate to the `example_notebooks/` directory.
2. Open the desired Jupyter Notebook (e.g., `01_ESR_eval.ipynb`).
3. Execute the cells. The notebooks are bound to local sample data situated in `example_notebooks/example_data`.

## Configuration

Standard analysis dictates adjustments to the global `config.yaml` located at the project root format. This file manages hardware channel mappings and remote data mounting points without requiring source code modifications.

```yaml
# 1. Base Directory Mounting
# Absolute paths to the main data directories per operating system.
data_folder_home:
  Windows: "G:\Bucherlab\Sensitivity_Optimization"
  Darwin: "/Volumes/001/Bucherlab/Sensitivity_Optimization"
  default: "./data"

# 2. Hardware Channel Indexing
# Defines the array indices corresponding to logic signals from the DAQ.
data_channels:
  reference: 0
  measurement: 1

# 3. Instrument Definitions
# Key definitions matching the YAML hardware logs. Defines target devices.
rf_calibration_device_key: "rf_source"
```

## Version Tracking

Maintain strict version control of software dependencies for reproducibility constraints.

The package instance provides version access natively:

```python
import evaluation_package
import evaluation_package.__version__ as version

print(f"Evaluation Package v{version}")
```

Plot titles and exported figures should list the package version utilized during the rendering frame to ensure historical compliance.
