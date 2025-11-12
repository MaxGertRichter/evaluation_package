# Evaluation Package

A Python package for evaluating nitrogen-vacancy (NV) center experiments in diamond. This package provides tools for analyzing standard quantum sensing measurements including ESR (Electron Spin Resonance), Rabi oscillations, CASR (Continuous AC Sensing and Ranging), and more.

## Features

### Experiment Types

- **ESR Analysis**: Extract resonance frequencies, peak detection, and spectral analysis
- **Rabi Oscillations**: Calculate π-pulse durations, contrast measurements, and coherence analysis
- **CASR Measurements**: Sensitivity calculations, FFT analysis, and AC field detection
- **DAQ Read**: Data acquisition and processing utilities

### Core Functionality

- Load and process experiment data from YAML configuration files and NumPy arrays
- Automated peak detection and fitting
- Light level and contrast calculations
- Fourier analysis for sensitivity measurements
- Parameter sweep generation and management
- Data handling for multi-measurement datasets
- Cross-platform support (Windows, macOS, Linux)

## Installation

### From source

```bash
git clone https://github.com/MaxGertRichter/evaluation_package.
cd evaluation_package
pip install -e .
```

## Version Tracking

To ensure reproducibility of your analysis, always track the package version in your evaluations.

### Accessing the Version

```python
import evaluation_package.__version__ as version
print(version)
```

### Best Practices

**1. Log version at the start of notebooks/scripts:**

```python
import evaluation_package

print(f"Evaluation Package v{version}")
```

**2. Include version in plot labels/captions:**

```python
import matplotlib.pyplot as plt
plt.figure()
plt.title(f'Analyzed with evaluation_package v{version}')
...
```

This practice ensures that results can be reproduced and any version-specific behavior can be traced.
