from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from src.baselines.dataset import (
    MultimodalSequenceDataset,
    collate_fn,
)

from src.baselines.b1_transformer.b1_model import (
    BaselineTransformer,
)


MODEL_CLASS = BaselineTransformer


TEST_PATH = Path(
    "data/processed/quantum/test.pkl"
)

MODEL_DIR = Path("models")

OUTPUT_DIR = Path(
    "data/results/b1"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 8


# ============================================================
# PARTICIPANT-LEVEL AGGREGATION

def aggregate_by_participant(
    probabilities,
    labels,
    participant_ids,
):

    probs = {}
    labs = {}

    for p, y, pid in zip(
        probabilities,
        labels,
        participant_ids,
    ):

        pid = str(pid)

        probs.setdefault(
            pid,
            [],
        ).append(float(p))

        labs.setdefault(
            pid,
            [],
        ).append(int(y))

    ids = sorted(probs)

    participant_probs = []
    participant_labels = []

    for pid in ids:

        ys = set(labs[pid])

        if len(ys) != 1:

            raise ValueError(
                f"Inconsistent labels "
                f"for participant "
                f"{pid}: {ys}"
            )

        participant_probs.append(
            float(
                np.mean(
                    probs[pid]
                )
            )
        )

        participant_labels.append(
            next(iter(ys))
        )

    return (
        np.asarray(
            participant_probs
        ),
        np.asarray(
            participant_labels
        ),
        ids,
    )


# ============================================================
# METRICS

def metrics(
    probabilities,
    labels,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    output = {
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

        output["roc_auc"] = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

    else:

        output["roc_auc"] = float(
            "nan"
        )

    return output


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL

def bootstrap_ci(
    probabilities,
    labels,
    threshold,
    metric_name,
    n=2000,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )

    values = []

    size = len(labels)

    for _ in range(n):

        idx = rng.integers(
            0,
            size,
            size=size,
        )

        y = labels[idx]
        p = probabilities[idx]

        if (
            metric_name == "roc_auc"
            and len(np.unique(y)) < 2
        ):
            continue

        values.append(
            metrics(
                p,
                y,
                threshold,
            )[metric_name]
        )

    if not values:

        return (
            float("nan"),
            float("nan"),
        )

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# ============================================================
# ARGUMENTS

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate B1_TRANSFORMER "
            "on held-out E-DAIC TEST."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Seed whose checkpoint "
            "should be evaluated."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed-specific checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR
        / f"b1_seed_{args.seed}_best.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Run the corresponding "
            "training seed first."
        )

    # --------------------------------------------------------
    # Seed-specific results directory
    # --------------------------------------------------------

    seed_results_dir = (
        OUTPUT_DIR
        / f"seed_{args.seed}"
    )

    seed_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("B1_TRANSFORMER TEST EVALUATION")
    print("=" * 70)

    print(
        f"Using device: {device}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Results directory: "
        f"{seed_results_dir}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_ds = MultimodalSequenceDataset(
        TEST_PATH
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    threshold = float(
        checkpoint["dev_threshold"]
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MODEL_CLASS().to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # TEST inference
    # --------------------------------------------------------

    records = []

    with torch.no_grad():

        for batch in test_loader:

            output = model(
                batch["audio"].to(device),
                batch["text"].to(device),
                batch["audio_mask"].to(device),
                batch["text_mask"].to(device),
            )

            probabilities = (
                torch.softmax(
                    output["logits"],
                    dim=-1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            labels = (
                batch["labels"]
                .numpy()
            )

            for i, probability in enumerate(
                probabilities
            ):

                records.append(
                    {
                        "participant_id":
                            batch[
                                "participant_ids"
                            ][i],

                        "utterance_id":
                            batch[
                                "utterance_ids"
                            ][i],

                        "probability":
                            float(
                                probability
                            ),

                        "label":
                            int(
                                labels[i]
                            ),

                        "phq8_total":
                            float(
                                batch[
                                    "phq8_totals"
                                ][i]
                            ),
                    }
                )

    # ========================================================
    # SEGMENT PREDICTIONS

    segment_path = (
        seed_results_dir
        / "segment_predictions.json"
    )

    segment_path.write_text(
        json.dumps(
            records,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Extract arrays
    # --------------------------------------------------------

    probabilities = np.asarray(
        [
            r["probability"]
            for r in records
        ]
    )

    labels = np.asarray(
        [
            r["label"]
            for r in records
        ]
    )

    participant_ids = [
        r["participant_id"]
        for r in records
    ]

    # --------------------------------------------------------
    # Participant-level aggregation
    # --------------------------------------------------------

    participant_probs, participant_labels, participant_ids = (
        aggregate_by_participant(
            probabilities,
            labels,
            participant_ids,
        )
    )

    # ========================================================
    # TEST METRICS
    # ========================================================

    test_metrics = metrics(
        participant_probs,
        participant_labels,
        threshold,
    )

    # --------------------------------------------------------
    # Bootstrap CIs
    # --------------------------------------------------------

    f1_ci = bootstrap_ci(
        participant_probs,
        participant_labels,
        threshold,
        "macro_f1",
        args.bootstrap,
        seed=args.seed,
    )

    auc_ci = bootstrap_ci(
        participant_probs,
        participant_labels,
        threshold,
        "roc_auc",
        args.bootstrap,
        seed=args.seed,
    )

    # ========================================================
    # PARTICIPANT PREDICTIONS

    participant_records = []

    for pid, probability, label in zip(
        participant_ids,
        participant_probs,
        participant_labels,
    ):

        participant_records.append(
            {
                "participant_id":
                    pid,

                "probability":
                    float(
                        probability
                    ),

                "prediction":
                    int(
                        probability
                        >= threshold
                    ),

                "label":
                    int(label),
            }
        )

    participant_path = (
        seed_results_dir
        / "participant_predictions.json"
    )

    participant_path.write_text(
        json.dumps(
            participant_records,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL TEST RESULTS

    results = {

        "model":
            "B1_TRANSFORMER",

        "checkpoint":
            str(checkpoint_path),

        "checkpoint_epoch":
            checkpoint.get("epoch"),

        "seed":
            checkpoint.get(
                "seed",
                args.seed,
            ),

        "dev_threshold":
            threshold,

        "test_records":
            len(records),

        "test_participants":
            len(participant_ids),

        "participant_level_metrics":
            test_metrics,

        "bootstrap_95_ci":
            {
                "macro_f1":
                    {
                        "lower":
                            f1_ci[0],

                        "upper":
                            f1_ci[1],
                    },

                "roc_auc":
                    {
                        "lower":
                            auc_ci[0],

                        "upper":
                            auc_ci[1],
                    },
            },

        "protocol":
            {
                "model_selection":
                    "DEV",

                "threshold_selection":
                    "DEV",

                "final_evaluation":
                    "held-out TEST",

                "aggregation":
                    "participant-level mean probability",
            },
    }

    results_path = (
        seed_results_dir
        / "test_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("B1_TRANSFORMER TEST RESULTS")
    print("=" * 70)

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Participants: "
        f"{len(participant_ids)}"
    )

    print(
        f"Segments: "
        f"{len(records)}"
    )

    print(
        f"DEV threshold: "
        f"{threshold:.3f}"
    )

    print()

    print(
        f"Macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro-F1 95% CI: "
        f"[{f1_ci[0]:.4f}, "
        f"{f1_ci[1]:.4f}]"
    )

    print(
        f"ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"ROC-AUC 95% CI: "
        f"[{auc_ci[0]:.4f}, "
        f"{auc_ci[1]:.4f}]"
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print()

    print(
        f"Segment predictions: "
        f"{segment_path}"
    )

    print(
        f"Participant predictions: "
        f"{participant_path}"
    )

    print(
        f"Test results: "
        f"{results_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()