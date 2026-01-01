"""
Configuration file for BEIR benchmarking and model fine-tuning.
"""
from pathlib import Path

# Base directory for benchmarking
BASE_DIR = Path(__file__).parent

# Model configuration
BASE_MODEL = "intfloat/e5-base-v2"
OPTIMIZED_MODEL_NAME = "ranksaga-optimized-e5-v2"

# BEIR datasets to benchmark
BEIR_DATASETS = ["scifact", "nfcorpus", "scidocs", "quora"]

# Training hyperparameters
TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "output_path": str(BASE_DIR / "models" / OPTIMIZED_MODEL_NAME),
}

# Paths
PATHS = {
    "models": BASE_DIR / "models",
    "results": BASE_DIR / "results",
    "baseline_results": BASE_DIR / "results" / "baseline",
    "optimized_results": BASE_DIR / "results" / "optimized",
    "datasets": BASE_DIR / "datasets",
}

# Evaluation metrics
EVALUATION_METRICS = [1, 3, 5, 10, 100, 1000]

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

