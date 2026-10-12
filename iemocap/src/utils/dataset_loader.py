"""
Creates PyTorch DataLoaders for IEMOCAP.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.dataset import IEMOCAPDataset


ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = (
    ROOT /
    "data" /
    "splits"
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

def get_dataloaders(

    batch_size=64,

    num_workers=0,

    pin_memory=True,

):

    train_dataset = IEMOCAPDataset(
        TRAIN_FILE
    )

    val_dataset = IEMOCAPDataset(
        VAL_FILE
    )

    test_dataset = IEMOCAPDataset(
        TEST_FILE
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=pin_memory,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,

    )
    
    print()

    print("=" * 60)

    print("Data Loaders")

    print("=" * 60)

    print()

    print(f"Train      : {len(train_dataset)}")

    print(f"Validation : {len(val_dataset)}")

    print(f"Test       : {len(test_dataset)}")

    print()

    print(f"Batch Size : {batch_size}")

    print()

    return (

        train_loader,

        val_loader,

        test_loader,

    )