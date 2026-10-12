from pathlib import Path
import pickle

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import nn

from tqdm import tqdm

from src.proposed.proposedModelV3 import (
    ProposedDiscordanceModelV3,
)


class ProposedDataset(Dataset):

    def __init__(
        self,
        samples,
    ):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        return {

            "audio": torch.tensor(
                sample["audio_embedding"],
                dtype=torch.float32,
            ),

            "text": torch.tensor(
                sample["text_embedding"],
                dtype=torch.float32,
            ),

            "label": torch.tensor(
                sample["label"],
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

DATA_DIR = Path(
    "data/processed"
)

with open(
    DATA_DIR / "hubertroberta_train.pkl",
    "rb"
) as f:

    train_data = pickle.load(f)

with open(
    DATA_DIR / "hubertroberta_val.pkl",
    "rb"
) as f:

    val_data = pickle.load(f)

print(
    f"Train samples: "
    f"{len(train_data)}"
)

# ---------------------------------------------------
# DATASETS
# ---------------------------------------------------

train_dataset = ProposedDataset(
    train_data
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {device}"
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = (
    ProposedDiscordanceModelV3()
    .to(device)
)

class_weights = torch.tensor(
    [1.0, 2.0]
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for batch in tqdm(train_loader):

        audio = (
            batch["audio"]
            .to(device)
        )

        text = (
            batch["text"]
            .to(device)
        )

        y = (
            batch["label"]
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(
            audio,
            text,
        )

        loss = criterion(
            outputs,
            y,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    print()
    print(
        f"Epoch {epoch+1}"
    )

    print(
        f"Train Loss: "
        f"{avg_loss:.4f}"
    )

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

torch.save(
    model.state_dict(),
    "proposed_discordance_v3.pt",
)

print()
print(
    "Proposed V3 model saved."
)