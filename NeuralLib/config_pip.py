from pathlib import Path

# Base directory for pip-installed package (user cache folder)
BASE_DIR = Path.home() / ".cache" / "NeuralLib"
DATA_BASE_DIR = BASE_DIR / "data"
RESULTS_BASE_DIR = BASE_DIR / "results"
HUGGING_MODELS_BASE_DIR = BASE_DIR / "hugging_prodmodels"

# Ensure necessary directories exist
for directory in [DATA_BASE_DIR, RESULTS_BASE_DIR, HUGGING_MODELS_BASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


