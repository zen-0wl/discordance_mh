from pathlib import Path

# ---------------------------------------------------
# project root
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------
# external dataset locations
# ---------------------------------------------------

EDAIC_DATASET_ROOT = Path(
    r"D:\Datasets\E-DAIC"
)

EDAIC_PROCESSED_ROOT = Path(
    r"D:\Datasets\E-DAIC-PROCESSED"
)

EDAIC_FEATURE_ROOT = Path(
    r"D:\Datasets\E-DAIC-FEATURES"
)


# ---------------------------------------------------
# internal repo directories
# ---------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"