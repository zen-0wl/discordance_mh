from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

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

from src.baselines.dataset import MultimodalSequenceDataset, collate_fn
from src.baselines.b1_transformer.b1_model import BaselineTransformer


MODEL_CLASS = BaselineTransformer

TRAIN_PATH = Path("data/processed/quantum/train.pkl")
DEV_PATH = Path("data/processed/quantum/dev.pkl")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("data/results/b1")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# REPRODUCIBILITY

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARTICIPANT-LEVEL AGGREGATION

def aggregate_by_participant(
    probabilities,
    labels,
    participant_ids,
):
    probs = {}
    labs = {}

    for p, y, pid in zip(probabilities, labels, participant_ids):
        pid = str(pid)

        probs.setdefault(pid, []).append(float(p))
        labs.setdefault(pid, []).append(int(y))

    ids = sorted(probs)

    out_p = []
    out_y = []

    for pid in ids:
        ys = set(labs[pid])

        if len(ys) != 1:
            raise ValueError(
                f"Inconsistent labels for participant {pid}: {ys}"
            )

        out_p.append(float(np.mean(probs[pid])))
        out_y.append(next(iter(ys)))

    return np.asarray(out_p), np.asarray(out_y), ids

# DEV THRESHOLD SELECTION
# ============================================================

def find_best_threshold(probabilities, labels):
    best_t = 0.50
    best_f1 = -float("inf")

    for t in np.linspace(0.05, 0.95, 181):
        pred = (probabilities >= t).astype(int)

        score = f1_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, best_f1

# METRICS
# ============================================================

def calculate_metrics(
    probabilities,
    labels,
    threshold,
):
    pred = (probabilities >= threshold).astype(int)

    out = {
        "macro_f1": float(
            f1_score(
                labels,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(labels, pred)
        ),
        "precision": float(
            precision_score(
                labels,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                pred,
                zero_division=0,
            )
        ),
    }

    if len(np.unique(labels)) == 2:
        out["roc_auc"] = float(
            roc_auc_score(labels, probabilities)
        )
    else:
        out["roc_auc"] = float("nan")

    return out

# DEV EVALUATION
# ============================================================

def evaluate_dev(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    labels = []
    participant_ids = []

    with torch.no_grad():

        for batch in loader:

            out = model(
                batch["audio"].to(device),
                batch["text"].to(device),
                batch["audio_mask"].to(device),
                batch["text_mask"].to(device),
            )

            probs = (
                torch.softmax(
                    out["logits"],
                    dim=-1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            probabilities.extend(
                probs.tolist()
            )

            labels.extend(
                batch["labels"]
                .numpy()
                .tolist()
            )

            participant_ids.extend(
                batch["participant_ids"]
            )

    return (
        np.asarray(probabilities),
        np.asarray(labels),
        participant_ids,
    )

# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train B1_TRANSFORMER."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    return parser.parse_args()

# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be > 0"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be > 0"
        )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(args.seed)

    device = get_device()

    print("=" * 70)
    print("B1_TRANSFORMER TRAINING")
    print("=" * 70)

    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    # --------------------------------------------------------
    # Seed-specific results directory
    # --------------------------------------------------------

    seed_results_dir = (
        RESULTS_DIR / f"seed_{args.seed}"
    )

    seed_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Seed results directory: "
        f"{seed_results_dir}"
    )
    print()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = MultimodalSequenceDataset(
        TRAIN_PATH
    )

    dev_ds = MultimodalSequenceDataset(
        DEV_PATH
    )

    print(
        f"Train records: {len(train_ds)}"
    )

    print(
        f"Dev records:   {len(dev_ds)}"
    )

    print()

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MODEL_CLASS().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_f1 = -float("inf")
    best_epoch = None
    best_threshold = 0.50

    # IMPORTANT:
    # Every seed gets its own checkpoint.
    best_path = (
        MODEL_DIR
        / f"b1_seed_{args.seed}_best.pt"
    )

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            audio = batch["audio"].to(device)
            text = batch["text"].to(device)

            audio_mask = batch[
                "audio_mask"
            ].to(device)

            text_mask = batch[
                "text_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                audio,
                text,
                audio_mask,
                text_mask,
            )

            loss = criterion(
                output["logits"],
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # ----------------------------------------------------
        # DEV evaluation
        # ----------------------------------------------------

        ps, ys, pids = evaluate_dev(
            model,
            dev_loader,
            device,
        )

        # Participant-level aggregation
        pp, py, pids = (
            aggregate_by_participant(
                ps,
                ys,
                pids,
            )
        )

        # DEV-only threshold selection
        threshold, _ = (
            find_best_threshold(
                pp,
                py,
            )
        )

        dev_metrics = calculate_metrics(
            pp,
            py,
            threshold,
        )

        record = {
            "epoch": epoch,
            "train_loss": (
                total_loss
                / len(train_loader)
            ),
            "dev_threshold": threshold,
            "dev_participants": len(py),
            **{
                f"dev_{k}": v
                for k, v in dev_metrics.items()
            },
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: "
            f"{record['train_loss']:.4f} | "
            f"Dev Macro-F1: "
            f"{dev_metrics['macro_f1']:.4f} | "
            f"Dev ROC-AUC: "
            f"{dev_metrics['roc_auc']:.4f} | "
            f"Threshold: "
            f"{threshold:.3f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint based ONLY on DEV Macro-F1
        # ----------------------------------------------------

        if (
            dev_metrics["macro_f1"]
            > best_f1
        ):

            best_f1 = (
                dev_metrics["macro_f1"]
            )

            best_epoch = epoch
            best_threshold = threshold

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "seed":
                    args.seed,

                "batch_size":
                    args.batch_size,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "dev_threshold":
                    threshold,

                "dev_metrics":
                    dev_metrics,

                "model_name":
                    "B1_TRANSFORMER",
            }

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"  -> New best checkpoint "
                f"saved: {best_path}"
            )

    # SAVE TRAINING HISTORY

    history_path = (
        seed_results_dir
        / "training_history.json"
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    # SAVE TRAINING SUMMARY

    summary = {
        "model": "B1_TRANSFORMER",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "best_dev_threshold": best_threshold,
        "checkpoint": str(best_path),
        "train_records": len(train_ds),
        "dev_records": len(dev_ds),
        "results_directory": str(
            seed_results_dir
        ),
    }

    summary_path = (
        seed_results_dir
        / "training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # FINAL OUTPUT

    print()
    print("=" * 70)
    print("B1 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"DEV-selected threshold: "
        f"{best_threshold:.3f}"
    )

    print(
        f"Checkpoint: "
        f"{best_path}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )

    print(
        f"Training summary: "
        f"{summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
