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

from src.proposed.dataset import (
    QuantumSequenceDataset,
    collate_fn,
)

from src.proposed.v5.model_v5 import (
    ProposedModelV5,
)


TEST_PATH = Path(
    "data/processed/quantum/test.pkl"
)

CHECKPOINT = Path(
    "models/v5_best.pt"
)

OUTPUT_DIR = Path(
    "data/results/v5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 8
RANK = 16


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
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:

    participant_probs: dict[
        str,
        list[float],
    ] = {}

    participant_labels: dict[
        str,
        list[int],
    ] = {}

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

    participant_probabilities = []
    participant_labels_out = []

    for participant_id in ids:

        participant_probabilities.append(
            float(
                np.mean(
                    participant_probs[
                        participant_id
                    ]
                )
            )
        )

        unique_labels = set(
            participant_labels[
                participant_id
            ]
        )

        if len(unique_labels) != 1:

            raise ValueError(
                "Inconsistent labels found for "
                f"participant {participant_id}: "
                f"{unique_labels}"
            )

        participant_labels_out.append(
            next(
                iter(
                    unique_labels
                )
            )
        )

    return (
        np.asarray(
            participant_probabilities,
            dtype=np.float64,
        ),
        np.asarray(
            participant_labels_out,
            dtype=np.int64,
        ),
        ids,
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

# Bootstrap 95% CI

def bootstrap_metric_ci(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:

    rng = np.random.default_rng(
        seed
    )

    n = len(labels)

    values = []

    for _ in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sampled_labels = labels[
            indices
        ]

        sampled_probabilities = (
            probabilities[
                indices
            ]
        )

        if (
            metric == "roc_auc"
            and len(
                np.unique(
                    sampled_labels
                )
            ) < 2
        ):

            continue

        sampled_metrics = calculate_metrics(
            sampled_probabilities,
            sampled_labels,
            threshold,
        )

        values.append(
            sampled_metrics[
                metric
            ]
        )

    if not values:

        return (
            float("nan"),
            float("nan"),
        )

    lower = float(
        np.percentile(
            values,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            values,
            97.5,
        )
    )

    return (
        lower,
        upper,
    )

# Argument parser

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ProposedModelV5 on "
            "the held-out E-DAIC test set."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{CHECKPOINT}\n"
            "Run train_v5.py first."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than 0."
        )

    device = get_device()

    # --------------------------------------------------------
    # Load test dataset

    test_dataset = QuantumSequenceDataset(
        TEST_PATH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Test records: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Load checkpoint

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    threshold = float(
        checkpoint[
            "dev_threshold"
        ]
    )

    print(
        f"Loaded checkpoint: "
        f"{CHECKPOINT}"
    )

    print(
        f"Best training epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Seed: "
        f"{checkpoint.get('seed', 'unknown')}"
    )

    print(
        f"DEV-selected threshold: "
        f"{threshold:.3f}"
    )


    model = ProposedModelV5(
        rank=RANK,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Inference

    segment_records = []

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
                .cpu()
                .numpy()
            )

            for i in range(
                len(probabilities)
            ):

                segment_records.append(
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
                                probabilities[i]
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

    # --------------------------------------------------------
    # Save segment predictions

    segment_path = (
        OUTPUT_DIR /
        "segment_predictions.json"
    )

    with open(
        segment_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            segment_records,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Convert to arrays

    segment_probabilities = np.asarray(
        [
            record["probability"]
            for record in segment_records
        ],
        dtype=np.float64,
    )

    segment_labels = np.asarray(
        [
            record["label"]
            for record in segment_records
        ],
        dtype=np.int64,
    )

    segment_participant_ids = [
        record["participant_id"]
        for record in segment_records
    ]

    # --------------------------------------------------------
    # Participant-level aggregation

    (
        participant_probabilities,
        participant_labels,
        participant_ids,
    ) = aggregate_by_participant(
        segment_probabilities,
        segment_labels,
        segment_participant_ids,
    )

    participant_metrics = calculate_metrics(
        participant_probabilities,
        participant_labels,
        threshold,
    )

    # --------------------------------------------------------
    # Bootstrap confidence intervals

    macro_f1_ci = bootstrap_metric_ci(
        participant_probabilities,
        participant_labels,
        threshold,
        metric="macro_f1",
        n_bootstrap=args.bootstrap,
        seed=42,
    )

    roc_auc_ci = bootstrap_metric_ci(
        participant_probabilities,
        participant_labels,
        threshold,
        metric="roc_auc",
        n_bootstrap=args.bootstrap,
        seed=42,
    )

    # --------------------------------------------------------
    # Participant predictions

    participant_predictions = []

    for participant_id, probability, label in zip(
        participant_ids,
        participant_probabilities,
        participant_labels,
    ):

        prediction = int(
            probability >= threshold
        )

        participant_predictions.append(
            {
                "participant_id":
                    participant_id,

                "probability":
                    float(
                        probability
                    ),

                "prediction":
                    prediction,

                "label":
                    int(label),
            }
        )

    participant_path = (
        OUTPUT_DIR /
        "participant_predictions.json"
    )

    with open(
        participant_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            participant_predictions,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final result

    results = {
        "model":
            "ProposedModelV5",

        "checkpoint":
            str(CHECKPOINT),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "seed":
            checkpoint.get(
                "seed"
            ),

        "dev_threshold":
            threshold,

        "test_records":
            len(segment_records),

        "test_participants":
            len(participant_ids),

        "participant_level_metrics":
            participant_metrics,

        "bootstrap_95_ci": {
            "macro_f1": {
                "lower":
                    macro_f1_ci[0],
                "upper":
                    macro_f1_ci[1],
            },

            "roc_auc": {
                "lower":
                    roc_auc_ci[0],
                "upper":
                    roc_auc_ci[1],
            },
        },

        "protocol": {
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
        OUTPUT_DIR /
        "test_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output

    print()
    print(
        "=" * 60
    )

    print(
        "V5 TEST EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Participants: "
        f"{len(participant_ids)}"
    )

    print(
        f"Segments: "
        f"{len(segment_records)}"
    )

    print(
        f"DEV threshold: "
        f"{threshold:.3f}"
    )

    print()

    print(
        f"Macro-F1: "
        f"{participant_metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro-F1 95% CI: "
        f"[{macro_f1_ci[0]:.4f}, "
        f"{macro_f1_ci[1]:.4f}]"
    )

    print()

    print(
        f"ROC-AUC: "
        f"{participant_metrics['roc_auc']:.4f}"
    )

    print(
        f"ROC-AUC 95% CI: "
        f"[{roc_auc_ci[0]:.4f}, "
        f"{roc_auc_ci[1]:.4f}]"
    )

    print()

    print(
        f"Accuracy: "
        f"{participant_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{participant_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{participant_metrics['recall']:.4f}"
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
        f"Results: "
        f"{results_path}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()