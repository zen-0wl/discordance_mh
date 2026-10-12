import pickle

from sklearn.model_selection import train_test_split

from src.config.settings import (
    ERISK_FEATURE_ROOT,
    SPLITS_DIR,
)

# ---------------------------------------------------
# configuration
# ---------------------------------------------------

RANDOM_STATE = 42

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

# ---------------------------------------------------
# paths
# ---------------------------------------------------

INPUT_FILE = (
    ERISK_FEATURE_ROOT /
    "embeddings.pkl"
)

TRAIN_FILE = (
    SPLITS_DIR /
    "train.pkl"
)

VAL_FILE = (
    SPLITS_DIR /
    "val.pkl"
)

TEST_FILE = (
    SPLITS_DIR /
    "test.pkl"
)

SPLITS_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

# ---------------------------------------------------
# load embeddings
# ---------------------------------------------------

with open(INPUT_FILE, "rb") as f:

    dataset = pickle.load(f)

labels = [

    sample["label"]

    for sample in dataset

]

# ---------------------------------------------------
# train / temp
# ---------------------------------------------------

train_set, temp_set = train_test_split(

    dataset,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=labels,

)

# ---------------------------------------------------
# val / test
# ---------------------------------------------------

temp_labels = [

    sample["label"]

    for sample in temp_set

]

val_set, test_set = train_test_split(

    temp_set,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=temp_labels,

)

# ---------------------------------------------------
# save
# ---------------------------------------------------

with open(TRAIN_FILE, "wb") as f:

    pickle.dump(train_set, f)

with open(VAL_FILE, "wb") as f:

    pickle.dump(val_set, f)

with open(TEST_FILE, "wb") as f:

    pickle.dump(test_set, f)

# ---------------------------------------------------
# statistics
# ---------------------------------------------------

print("=" * 60)

print("Dataset Split")

print("=" * 60)

print()

print(f"Train      : {len(train_set)}")

print(f"Validation : {len(val_set)}")

print(f"Test       : {len(test_set)}")

print()

print("Saved to")

print(TRAIN_FILE)

print(VAL_FILE)

print(TEST_FILE)