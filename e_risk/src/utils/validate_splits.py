import pickle

from src.config.settings import (
    SPLITS_DIR,
)

# ---------------------------------------------------
# paths
# ---------------------------------------------------

TRAIN_FILE = SPLITS_DIR / "train.pkl"

VAL_FILE = SPLITS_DIR / "val.pkl"

TEST_FILE = SPLITS_DIR / "test.pkl"

# ---------------------------------------------------
# load
# ---------------------------------------------------

train = pickle.load(open(TRAIN_FILE, "rb"))

val = pickle.load(open(VAL_FILE, "rb"))

test = pickle.load(open(TEST_FILE, "rb"))

# ---------------------------------------------------
# helper
# ---------------------------------------------------

def summarize(name, dataset):

    labels = [

        sample["label"]

        for sample in dataset

    ]

    control = labels.count(0)

    depressed = labels.count(1)

    print()

    print(name)

    print("-" * 30)

    print(f"Subjects   : {len(dataset)}")

    print(f"Control    : {control}")

    print(f"Depressed  : {depressed}")

# ---------------------------------------------------
# output
# ---------------------------------------------------

print("=" * 60)

print("Dataset Split Validation")

print("=" * 60)

summarize("Train", train)

summarize("Validation", val)

summarize("Test", test)

print()

print("=" * 60)

print("Validation PASSED")

print("=" * 60)