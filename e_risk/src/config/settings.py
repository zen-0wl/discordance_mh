from pathlib import Path

# ---------------------------------------------------
# project root
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------
# external dataset locations
# ---------------------------------------------------

ERISK_DATASET_ROOT = Path(
    r"D:\Datasets\ERISK-T2"
)

ERISK_PROCESSED_ROOT = Path(
    r"D:\Datasets\ERISK-PROCESSED"
)

ERISK_FEATURE_ROOT = Path(
    r"D:\Datasets\ERISK-FEATURES"
)

# ---------------------------------------------------
# internal repo directories
# ---------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SPLITS_DIR = DATA_DIR / "splits"

MODEL_DIR = PROJECT_ROOT / "models"

B1_MODEL_DIR = MODEL_DIR / "b1_transformer"

B2_MODEL_DIR = MODEL_DIR / "b2_attention"

B3_MODEL_DIR = MODEL_DIR / "b3_quantumhead"

PROPOSED_MODEL_DIR = MODEL_DIR / "proposed"

B1_RESULTS_PATH = B1_MODEL_DIR / "results.json"

B2_RESULTS_PATH = B2_MODEL_DIR / "results.json"

B3_RESULTS_PATH = B3_MODEL_DIR / "results.json"

PROPOSED_RESULTS_PATH = PROPOSED_MODEL_DIR / "results.json"

for directory in [

    MODEL_DIR,

    B1_MODEL_DIR,

    B2_MODEL_DIR,

    B3_MODEL_DIR,

    PROPOSED_MODEL_DIR,

]:

    directory.mkdir(

        parents=True,

        exist_ok=True,

    )