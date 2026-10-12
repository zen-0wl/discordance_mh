from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.proposed.dataset import (
    QuantumSequenceDataset,
    collate_fn,
)

from src.proposed.v4.model_v4 import (
    ProposedModelV4,
)


TRAIN_PATH = Path(
    "data/processed/quantum/train.pkl"
)

DEV_PATH = Path(
    "data/processed/quantum/dev.pkl"
)

MODEL_DIR = Path(
    "models/"
)

RESULTS_DIR = Path(
    "data/results/v4"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANK = 16

# Reproducibility

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    return device

# Participant-level aggregation

def aggregate_by_participant(
    probabilities: np.ndarray,
    labels: np.ndarray,
    participant_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Aggregate utterance/segment predictions to participant level.

    Probability:
        mean probability across that participant's records.

    Label:
        participant label.

    The official E-DAIC split is participant-level, so all
    screening metrics used for model selection are calculated
    after participant-level aggregation.
    """

    participant_probs: dict[str, list[float]] = {}
    participant_labels: dict[str, list[int]] = {}

    for probability, label, participant_id in zip(
        probabilities,
        labels,
        participant_ids,
    ):

        participant_id = str(
            participant_id
        )

        participant_probs.setdefault(
            participant_id,
            [],
        )

        participant_labels.setdefault(
            participant_id,
            [],
        )

        participant_probs[
            participant_id
        ].append(
            float(probability)
        )

        participant_labels[
            participant_id
        ].append(
            int(label)
        )

    ids = sorted(
        participant_probs.keys()
    )

    probabilities_out = []
    labels_out = []

    for participant_id in ids:

        probabilities_out.append(
            float(
                np.mean(
                    participant_probs[
                        participant_id
                    ]
                )
            )
        )

        labels_for_participant = set(
            participant_labels[
                participant_id
            ]
        )

        if len(labels_for_participant) != 1:

            raise ValueError(
                "Inconsistent labels found for "
                f"participant {participant_id}: "
                f"{labels_for_participant}"
            )

        labels_out.append(
            next(
                iter(
                    labels_for_participant
                )
            )
        )

    return (
        np.asarray(
            probabilities_out,
            dtype=np.float64,
        ),
        np.asarray(
            labels_out,
            dtype=np.int64,
        ),
        ids,
    )

# Threshold selection

def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Select the threshold that maximizes participant-level
    Macro-F1.

    Threshold is selected ONLY on DEV.
    """

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    best_threshold = 0.50
    best_macro_f1 = -float("inf")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_macro_f1,
    )

# Metrics

def calculate_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
    }

    if len(np.unique(labels)) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

    else:

        metrics["roc_auc"] = float(
            "nan"
        )

    return metrics

# DEV inference

def evaluate_dev(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:

    model.eval()

    probabilities = []
    labels = []
    participant_ids = []

    with torch.no_grad():

        for batch in loader:

            audio = (
                batch["audio"]
                .to(device)
            )

            text = (
                batch["text"]
                .to(device)
            )

            audio_mask = (
                batch["audio_mask"]
                .to(device)
            )

            text_mask = (
                batch["text_mask"]
                .to(device)
            )

            output = model(
                audio,
                text,
                audio_mask,
                text_mask,
            )

            batch_probabilities = (
                torch.softmax(
                    output["logits"],
                    dim=-1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            batch_labels = (
                batch["labels"]
                .cpu()
                .numpy()
            )

            probabilities.extend(
                batch_probabilities.tolist()
            )

            labels.extend(
                batch_labels.tolist()
            )

            participant_ids.extend(
                batch["participant_ids"]
            )

    return (
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        participant_ids,
    )

# Argument parser

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train ProposedModelV4 with "
            "participant-level DEV model selection."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training/evaluation batch size.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be greater than 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than 0."
        )

    set_seed(
        args.seed
    )

    device = get_device()

    print()

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print()

    # --------------------------------------------------------
    # Dataset

    train_dataset = QuantumSequenceDataset(
        TRAIN_PATH
    )

    dev_dataset = QuantumSequenceDataset(
        DEV_PATH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Train records: {len(train_dataset)}"
    )

    print(
        f"Dev records:   {len(dev_dataset)}"
    )

    print()

    # --------------------------------------------------------
    # Model

    model = ProposedModelV4(
        rank=RANK,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Checkpoint

    best_dev_macro_f1 = -float(
        "inf"
    )

    best_epoch = None
    best_threshold = 0.50

    best_path = (
        MODEL_DIR /
        "v4_best.pt"
    )

    history = []

    # --------------------------------------------------------
    # Training

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            audio = (
                batch["audio"]
                .to(device)
            )

            text = (
                batch["text"]
                .to(device)
            )

            audio_mask = (
                batch["audio_mask"]
                .to(device)
            )

            text_mask = (
                batch["text_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                audio,
                text,
                audio_mask,
                text_mask,
            )

            logits = output[
                "logits"
            ]

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss /
            len(train_loader)
        )

        # ----------------------------------------------------
        # DEV evaluation
        (
            dev_segment_probabilities,
            dev_segment_labels,
            dev_segment_participants,
        ) = evaluate_dev(
            model,
            dev_loader,
            device,
        )

        (
            dev_probabilities,
            dev_labels,
            dev_participant_ids,
        ) = aggregate_by_participant(
            dev_segment_probabilities,
            dev_segment_labels,
            dev_segment_participants,
        )

        (
            dev_threshold,
            dev_macro_f1,
        ) = find_best_threshold(
            dev_probabilities,
            dev_labels,
        )

        dev_metrics = calculate_metrics(
            dev_probabilities,
            dev_labels,
            dev_threshold,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(
                average_loss
            ),
            "dev_threshold": float(
                dev_threshold
            ),
            "dev_macro_f1": float(
                dev_metrics["macro_f1"]
            ),
            "dev_roc_auc": float(
                dev_metrics["roc_auc"]
            ),
            "dev_accuracy": float(
                dev_metrics["accuracy"]
            ),
            "dev_precision": float(
                dev_metrics["precision"]
            ),
            "dev_recall": float(
                dev_metrics["recall"]
            ),
            "dev_participants": int(
                len(dev_labels)
            ),
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:02d} "
            f"| Train loss: {average_loss:.4f} "
            f"| Dev Macro-F1: "
            f"{dev_metrics['macro_f1']:.4f} "
            f"| Dev ROC-AUC: "
            f"{dev_metrics['roc_auc']:.4f} "
            f"| Threshold: "
            f"{dev_threshold:.3f}"
        )

        # Best-model selection

        if (
            dev_metrics["macro_f1"]
            > best_dev_macro_f1
        ):

            best_dev_macro_f1 = (
                dev_metrics["macro_f1"]
            )

            best_epoch = epoch
            best_threshold = (
                dev_threshold
            )

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "seed":
                    args.seed,

                "rank":
                    RANK,

                "batch_size":
                    args.batch_size,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "dev_threshold":
                    float(
                        dev_threshold
                    ),

                "dev_metrics":
                    dev_metrics,

                "model_name":
                    "ProposedModelV4",
            }

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"  -> New best checkpoint "
                f"saved: {best_path}"
            )

    # Save training history

    history_path = (
        RESULTS_DIR /
        f"training_history_seed_{args.seed}.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # Save training summary

    summary = {
        "model": "ProposedModelV4",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": float(
            best_dev_macro_f1
        ),
        "best_dev_threshold": float(
            best_threshold
        ),
        "checkpoint": str(
            best_path
        ),
        "train_records": len(
            train_dataset
        ),
        "dev_records": len(
            dev_dataset
        ),
        "dev_participants": len(
            dev_dataset
        ),
    }

    summary_path = (
        RESULTS_DIR /
        f"training_summary_seed_{args.seed}.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "Training complete."
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{best_dev_macro_f1:.4f}"
    )

    print(
        f"DEV-selected threshold: "
        f"{best_threshold:.3f}"
    )

    print(
        f"Checkpoint: {best_path}"
    )

    print(
        f"History: {history_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()