from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
SPLITS_DATA = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42