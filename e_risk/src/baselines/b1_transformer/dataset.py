from src.config.settings import (
    SPLITS_DIR,
)

from src.utils.dataset_loader import (
    create_dataloader,
)

# ---------------------------------------------------
# paths
# ---------------------------------------------------

TRAIN_FILE = SPLITS_DIR / "train.pkl"

VAL_FILE = SPLITS_DIR / "val.pkl"

TEST_FILE = SPLITS_DIR / "test.pkl"


# ---------------------------------------------------
# dataloaders
# ---------------------------------------------------

def get_train_loader(

    batch_size=16,

    shuffle=True,

):

    return create_dataloader(

        TRAIN_FILE,

        batch_size=batch_size,

        shuffle=shuffle,

    )


def get_validation_loader(

    batch_size=16,

):

    return create_dataloader(

        VAL_FILE,

        batch_size=batch_size,

        shuffle=False,

    )


def get_test_loader(

    batch_size=16,

):

    return create_dataloader(

        TEST_FILE,

        batch_size=batch_size,

        shuffle=False,

    )