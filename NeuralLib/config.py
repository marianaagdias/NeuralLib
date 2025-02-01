import os

"""
Configuration file for setting up directories and paths used throughout the project.

This version dynamically adjusts the directory structure based on the location of the 
config file (which is inside `NeuralLib`). It assumes the following structure relative to the `dev` folder:
- `dev` contains `NeuralLib` (the library), `data` (datasets), `results`, and `pretrained_models`.

The paths adjust automatically based on the directory structure, avoiding the need for 
hardcoding paths in different environments.
"""

# Base directory for the `dev` folder (assumes `config.py` is inside NeuralLib in the `dev` folder)
DEV_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_BASE_DIR = os.path.join(DEV_BASE_DIR, "data")
RESULTS_BASE_DIR = os.path.join(DEV_BASE_DIR, "results")
HUGGING_MODELS_BASE_DIR = os.path.join(DEV_BASE_DIR, "hugging_prodmodels")

# Paths to ECG data files
ECG_DATA_PATH = os.path.join(DATA_BASE_DIR, "gib01_ecg", "ecg_sr1440.npz")
PEAKS_DATA_PATH = os.path.join(DATA_BASE_DIR, "gib01_ecg", "peaks.json")

# Directories for saving processed datasets
DATASETS_GIB01 = os.path.join(DATA_BASE_DIR, "gib01_ecg", "datasets")

# Create the directories if they don't exist
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(HUGGING_MODELS_BASE_DIR, exist_ok=True)

