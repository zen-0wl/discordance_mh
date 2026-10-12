from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MultimodalSequenceDataset(Dataset):
    """Shared sequence dataset for the three classical baseline models.

    Reads the Stage C HuBERT + DeBERTa sequence-level PKLs.
    The loader is model/representation agnostic: no projection, attention,
    fusion, density-state construction, or quantum operations are performed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        with open(self.path, "rb") as f:
            self.records = pickle.load(f)
        if not isinstance(self.records, list):
            raise TypeError(
                "Expected dataset to be a list of records, "
                f"got {type(self.records).__name__}"
            )
        if not self.records:
            raise ValueError(f"Dataset is empty: {self.path}")
        self._validate_records()

    def _validate_records(self) -> None:
        required_fields = {
            "participant_id", "utterance_id", "audio_hidden_states",
            "text_hidden_states", "label", "phq8_total", "split",
        }
        for index, record in enumerate(self.records):
            missing = required_fields - set(record.keys())
            if missing:
                raise KeyError(
                    f"Record {index} is missing required fields: {sorted(missing)}"
                )
            if not record.get("participant_id"):
                raise ValueError(f"Record {index} has no participant_id.")
            if not record.get("utterance_id"):
                raise ValueError(f"Record {index} has no utterance_id.")

            audio = torch.as_tensor(record["audio_hidden_states"], dtype=torch.float32)
            text = torch.as_tensor(record["text_hidden_states"], dtype=torch.float32)

            if audio.ndim != 2 or audio.shape[0] == 0:
                raise ValueError(
                    f"Record {index} audio_hidden_states must be non-empty 2-D; "
                    f"got {tuple(audio.shape)}"
                )
            if text.ndim != 2 or text.shape[0] == 0:
                raise ValueError(
                    f"Record {index} text_hidden_states must be non-empty 2-D; "
                    f"got {tuple(text.shape)}"
                )
            if audio.shape[1] != 768:
                raise ValueError(
                    f"Record {index} audio feature dimension must be 768, "
                    f"got {audio.shape[1]}"
                )
            if text.shape[1] != 768:
                raise ValueError(
                    f"Record {index} text feature dimension must be 768, "
                    f"got {text.shape[1]}"
                )
            if not torch.isfinite(audio).all():
                raise ValueError(f"Record {index} contains non-finite audio values.")
            if not torch.isfinite(text).all():
                raise ValueError(f"Record {index} contains non-finite text values.")

            label = int(record["label"])
            if label not in (0, 1):
                raise ValueError(f"Record {index} has invalid label: {label}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "audio": torch.as_tensor(record["audio_hidden_states"], dtype=torch.float32),
            "text": torch.as_tensor(record["text_hidden_states"], dtype=torch.float32),
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "participant_id": str(record["participant_id"]),
            "utterance_id": str(record["utterance_id"]),
            "phq8_total": float(record["phq8_total"]),
            "split": str(record["split"]),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length HuBERT/DeBERTa sequences and return valid-position masks."""
    if not batch:
        raise ValueError("collate_fn received an empty batch.")

    audio_sequences = [item["audio"] for item in batch]
    text_sequences = [item["text"] for item in batch]

    audio_lengths = torch.tensor(
        [x.shape[0] for x in audio_sequences], dtype=torch.long
    )
    text_lengths = torch.tensor(
        [x.shape[0] for x in text_sequences], dtype=torch.long
    )

    audio = pad_sequence(audio_sequences, batch_first=True, padding_value=0.0)
    text = pad_sequence(text_sequences, batch_first=True, padding_value=0.0)

    audio_mask = (
        torch.arange(audio.shape[1], dtype=torch.long).unsqueeze(0)
        < audio_lengths.unsqueeze(1)
    )
    text_mask = (
        torch.arange(text.shape[1], dtype=torch.long).unsqueeze(0)
        < text_lengths.unsqueeze(1)
    )

    return {
        "audio": audio,
        "text": text,
        "audio_mask": audio_mask,
        "text_mask": text_mask,
        "audio_lengths": audio_lengths,
        "text_lengths": text_lengths,
        "labels": torch.stack([item["label"] for item in batch]),
        "participant_ids": [item["participant_id"] for item in batch],
        "utterance_ids": [item["utterance_id"] for item in batch],
        "phq8_totals": torch.tensor(
            [item["phq8_total"] for item in batch], dtype=torch.float32
        ),
        "splits": [item["split"] for item in batch],
    }


def create_dataloader(
    dataset_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Convenience DataLoader factory shared by all three baselines."""
    from torch.utils.data import DataLoader

    dataset = MultimodalSequenceDataset(dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


if __name__ == "__main__":
    print("Shared baseline multimodal sequence dataset test")
    print("-" * 60)
    paths = {
        "train": Path("data/processed/quantum/train.pkl"),
        "dev": Path("data/processed/quantum/dev.pkl"),
        "test": Path("data/processed/quantum/test.pkl"),
    }

    for split, path in paths.items():
        dataset = MultimodalSequenceDataset(path)
        first = dataset[0]
        participants = {str(r["participant_id"]) for r in dataset.records}
        print(
            f"{split:>5}: {len(dataset):>6} records | {len(participants):>3} participants | "
            f"participant={first['participant_id']} | utterance={first['utterance_id']} | "
            f"audio={tuple(first['audio'].shape)} | text={tuple(first['text'].shape)} | "
            f"label={first['label'].item()}"
        )

    dataset = MultimodalSequenceDataset(paths["train"])
    batch = collate_fn([dataset[0], dataset[1]])
    print("-" * 60)
    print("Audio batch:", tuple(batch["audio"].shape))
    print("Text batch:", tuple(batch["text"].shape))
    print("Audio mask:", tuple(batch["audio_mask"].shape))
    print("Text mask:", tuple(batch["text_mask"].shape))
    print("Labels:", batch["labels"].tolist())
    print("Dataset test: PASS")
