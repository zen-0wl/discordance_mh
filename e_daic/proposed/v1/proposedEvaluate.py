from pathlib import Path
import pickle

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.proposed.proposedModel import (
    ProposedDiscordanceModel,
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
    DATA_DIR / "hubertroberta_test.pkl",
    "rb"
) as f:

    test_data = pickle.load(f)

print(
    f"Test samples: "
    f"{len(test_data)}"
)

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

test_dataset = ProposedDataset(
    test_data
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
)

# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = (
    ProposedDiscordanceModel()
    .to(device)
)

model.load_state_dict(
    torch.load(
        "proposed_discordance.pt",
        map_location=device,
    )
)

model.eval()

print("Model loaded.")

# ---------------------------------------------------
# INFERENCE
# ---------------------------------------------------

predictions = []
targets = []

with torch.no_grad():

    for batch in test_loader:

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

        outputs = model(
            audio,
            text,
        )

        preds = torch.argmax(
            outputs,
            dim=1,
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            y.cpu().numpy()
        )

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

print()
print(
    classification_report(
        targets,
        predictions,
    )
)

print()
print(
    "Accuracy:",
    accuracy_score(
        targets,
        predictions,
    )
)

print(
    "Precision:",
    precision_score(
        targets,
        predictions,
    )
)

print(
    "Recall:",
    recall_score(
        targets,
        predictions,
    )
)

print(
    "F1:",
    f1_score(
        targets,
        predictions,
    )
)

print(
    "Macro-F1:",
    f1_score(
        targets,
        predictions,
        average="macro",
    )
)

print()
print(
    confusion_matrix(
        targets,
        predictions,
    )
)