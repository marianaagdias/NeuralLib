import os

"""
Configuration file for setting up directories and paths used throughout the project.

You can change or add directories here to specify where the data is stored (DATA_BASE_DIR),
where the code, results, and other files should be saved (DEV_BASE_DIR), and any other 
main directories required for the project.

- DATA_BASE_DIR: This is the base directory where your raw and processed datasets are stored.
- DEV_BASE_DIR: This is the base directory where the code, results, and trained models are stored.

The results of the models will be saved in the DEV_BASE_DIR, while the processed datasets 
split for training, testing, etc., will be saved in the DATA_BASE_DIR.

Feel free to adjust the paths according to your local setup.
"""

run_ = ['Turing', 'laptop', 'Ada']
run = run_[1]  # change here to select your current environment

# Base directory where your data is stored
if run == 'laptop':
    DEV_BASE_DIR = "C:/Users/Catia Bastos/dev"
    DATA_BASE_DIR = "C:/Users/Catia Bastos/dev/data"
elif run == 'Turing':
    DEV_BASE_DIR = ""  # change here!!
    DATA_BASE_DIR = ""
elif run == 'Ada':
    DEV_BASE_DIR = "C:/Users/Catia Bastos/dev"
    DATA_BASE_DIR = "C:/Users/Catia Bastos/dev/data"
else:
    DEV_BASE_DIR = ""  # change here!!
    DATA_BASE_DIR = ""

# Paths to ECG data files
ECG_DATA_PATH = os.path.join(DATA_BASE_DIR, "gib01_ecg", "ecg_sr1440.npz")
PEAKS_DATA_PATH = os.path.join(DATA_BASE_DIR, "gib01_ecg", "peaks.json")

# Directories for saving processed datasets
DATASETS_GIB01 = os.path.join(DATA_BASE_DIR, "gib01_ecg", "datasets")

# Directories for saving models results
RESULTS_PEAK_DETECTION = os.path.join(DEV_BASE_DIR, "results", "peak_detection")

# Directories for saving trained models
TRAINED_MODELS = os.path.join(DEV_BASE_DIR, "pretrained_models")

# Create the directories if they don't exist
os.makedirs(DATASETS_GIB01, exist_ok=True)
os.makedirs(RESULTS_PEAK_DETECTION, exist_ok=True)
os.makedirs(TRAINED_MODELS, exist_ok=True)
