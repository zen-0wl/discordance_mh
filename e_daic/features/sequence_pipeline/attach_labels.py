from pathlib import Path
import pickle

import pandas as pd


MANIFEST_PATH = (
    Path("data")
    / "processed"
    / "official_split"
    / "participant_manifest.csv"
)

QUANTUM_DIR = (
    Path("data")
    / "processed"
    / "quantum"
)

# Load official participant manifest
# ============================================================

print("=" * 70)
print("Loading official participant manifest")
print("=" * 70)

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "participant_id",
    "split",
    "phq8_total",
    "label",
}

missing = required_columns - set(manifest.columns)

if missing:
    raise ValueError(
        f"Manifest is missing required columns: {sorted(missing)}"
    )


# Normalise participant IDs to strings so they match the PKLs.
manifest["participant_id"] = (
    manifest["participant_id"]
    .astype(str)
    .str.strip()
)


# Make sure there is exactly one official record per participant.
if manifest["participant_id"].duplicated().any():
    duplicates = (
        manifest.loc[
            manifest["participant_id"].duplicated(keep=False),
            "participant_id",
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        f"Duplicate participant IDs in official manifest: {duplicates}"
    )


official_metadata = (
    manifest[
        [
            "participant_id",
            "split",
            "phq8_total",
            "label",
        ]
    ]
    .set_index("participant_id")
)


print(
    f"Official participants: {len(official_metadata)}"
)

# Process each quantum split
# ============================================================

for split in ["train", "dev", "test"]:

    path = QUANTUM_DIR / f"{split}.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Quantum dataset not found: {path}"
        )

    print()
    print("=" * 70)
    print(f"Processing {split}.pkl")
    print("=" * 70)

    with open(path, "rb") as f:
        records = pickle.load(f)

    print(f"Records loaded: {len(records)}")

    # --------------------------------------------------------
    # Validate participant IDs
    # --------------------------------------------------------

    record_participants = {
        str(record["participant_id"]).strip()
        for record in records
    }

    missing_participants = (
        record_participants
        - set(official_metadata.index)
    )

    if missing_participants:
        raise ValueError(
            f"{split}: participants missing from official "
            f"manifest: {sorted(missing_participants)}"
        )

    # --------------------------------------------------------
    # Attach authoritative metadata
    # --------------------------------------------------------

    for record in records:

        pid = str(
            record["participant_id"]
        ).strip()

        metadata = official_metadata.loc[pid]

        official_split = str(
            metadata["split"]
        ).strip()

        # Verify existing split agrees with official split.
        existing_split = str(
            record["split"]
        ).strip()

        if existing_split != official_split:
            raise ValueError(
                f"Split mismatch for participant {pid}: "
                f"record={existing_split}, "
                f"official={official_split}"
            )

        record["phq8_total"] = int(
            metadata["phq8_total"]
        )

        record["label"] = int(
            metadata["label"]
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    labels = {
        int(record["label"])
        for record in records
    }

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{split}: unexpected labels: {labels}"
        )

    # --------------------------------------------------------
    # Save updated dataset
    # --------------------------------------------------------

    with open(path, "wb") as f:
        pickle.dump(
            records,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    participants = {
        str(record["participant_id"])
        for record in records
    }

    positive = sum(
        int(record["label"]) == 1
        for record in records
    )

    negative = sum(
        int(record["label"]) == 0
        for record in records
    )

    print(
        f"Participants: {len(participants)}"
    )

    print(
        f"Records:      {len(records)}"
    )

    print(
        f"Label 0:      {negative}"
    )

    print(
        f"Label 1:      {positive}"
    )

    print(
        f"Saved:        {path}"
    )


print()
print("=" * 70)
print("OFFICIAL METADATA ATTACHMENT COMPLETE")
print("=" * 70)