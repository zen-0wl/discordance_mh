"""
Splitting is performed by dialogue
(Session + Dialogue) to prevent
conversation leakage.
"""

from pathlib import Path
import pickle
import random
from collections import defaultdict


RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

SPLITS_DIR = DATA_DIR / "splits"

DATASET_FILE = PROCESSED_DIR / "dataset.pkl"

TRAIN_FILE = SPLITS_DIR / "train.pkl"

VAL_FILE = SPLITS_DIR / "val.pkl"

TEST_FILE = SPLITS_DIR / "test.pkl"

SPLITS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_dataset():

    with open(DATASET_FILE, "rb") as f:

        dataset = pickle.load(f)

    print(
        f"Loaded dataset: {len(dataset)} samples."
    )

    return dataset


def group_by_dialogue(dataset):

    grouped = defaultdict(list)

    for sample in dataset:

        key = (
            sample["session"],
            sample["dialogue"],
        )

        grouped[key].append(sample)

    return grouped


# --------------------------------------------------
# split dialogue ids
# --------------------------------------------------

def split_dialogues(dialogue_ids):

    random.seed(RANDOM_SEED)

    dialogue_ids = list(dialogue_ids)

    random.shuffle(dialogue_ids)

    total = len(dialogue_ids)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(total * VAL_RATIO)
    )

    train_ids = dialogue_ids[:train_end]

    val_ids = dialogue_ids[
        train_end:val_end
    ]

    test_ids = dialogue_ids[val_end:]

    return (
        train_ids,
        val_ids,
        test_ids,
    )

def build_split(
    grouped_data,
    dialogue_ids,
):

    split = []

    for key in dialogue_ids:

        split.extend(
            grouped_data[key]
        )

    return split

def save_split(
    split_data,
    output_file,
):

    with open(output_file, "wb") as f:

        pickle.dump(
            split_data,
            f,
        )

# --------------------------------------------------
# statistics
# --------------------------------------------------

def print_statistics(
    train,
    val,
    test,
):

    total = (
        len(train)
        + len(val)
        + len(test)
    )

    print()

    print("=" * 60)

    print("Dataset Split")

    print("=" * 60)

    print()

    print(
        f"Train      : {len(train):5d} "
        f"({len(train)/total:.1%})"
    )

    print(
        f"Validation : {len(val):5d} "
        f"({len(val)/total:.1%})"
    )

    print(
        f"Test       : {len(test):5d} "
        f"({len(test)/total:.1%})"
    )

    print()

    print("Saved to")

    print(TRAIN_FILE)

    print(VAL_FILE)

    print(TEST_FILE)

if __name__ == "__main__":

    dataset = load_dataset()

    grouped = group_by_dialogue(
        dataset
    )

    dialogue_ids = sorted(
        grouped.keys()
    )

    (
        train_ids,
        val_ids,
        test_ids,
    ) = split_dialogues(
        dialogue_ids
    )

    train_set = build_split(
        grouped,
        train_ids,
    )

    val_set = build_split(
        grouped,
        val_ids,
    )

    test_set = build_split(
        grouped,
        test_ids,
    )

    save_split(
        train_set,
        TRAIN_FILE,
    )

    save_split(
        val_set,
        VAL_FILE,
    )

    save_split(
        test_set,
        TEST_FILE,
    )

    print_statistics(
        train_set,
        val_set,
        test_set,
    )

    print()

    print(
        "IEMOCAP dataset splitting complete."
    )