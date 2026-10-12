"""
Loads train/val/test pickle files and returns
audio embeddings, text embeddings and labels.
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from src.config.emotion_config import (
    EMOTION_TO_ID,
    NUM_CLASSES,
)

class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        with open(self.dataset_path, "rb") as f:

            data = pickle.load(f)

        self.samples = []

        skipped = 0

        for sample in data:

            emotion = sample["emotion"]

            if emotion not in EMOTION_TO_ID:

                skipped += 1

                continue

            self.samples.append(sample)

        print()

        print(
            f"Loaded {len(self.samples)} samples."
        )

        print(
            f"Skipped {skipped} unsupported samples."
        )

    # --------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        audio = torch.tensor(

            sample["audio_embedding"],

            dtype=torch.float32,

        )

        text = torch.tensor(

            sample["text_embedding"],

            dtype=torch.float32,

        )

        label = torch.tensor(

            EMOTION_TO_ID[
                sample["emotion"]
            ],

            dtype=torch.long,

        )

        return {

            "audio": audio,

            "text": text,

            "label": label,

            "emotion": sample["emotion"],

            "session": sample["session"],

            "dialogue": sample["dialogue"],

            "speaker": sample["speaker"],

        }


# --------------------------------------------------
# utility
# --------------------------------------------------

def get_num_classes():

    return NUM_CLASSES