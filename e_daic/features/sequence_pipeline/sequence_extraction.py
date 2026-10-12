"""
Sequence-level HuBERT + DeBERTa extraction

Creates a separate sequence-level dataset for the corrected
quantum-inspired representation.

Existing outputs are NOT modified.

Each record contains:
    - audio_hidden_states: [T_audio, 768]
    - text_hidden_states:  [T_text, 768]
    - audio_embedding:     [768]  (retained for compatibility)
    - text_embedding:      [768]  (retained for compatibility)
    - participant / utterance metadata

The hidden-state sequences are taken directly from:
    HuBERT outputs.last_hidden_state
    DeBERTa outputs.last_hidden_state

No artificial reshaping or repetition of pooled embeddings is used.
"""

from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
    AutoModel,
)

# Configuration

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HUBERT_NAME = "facebook/hubert-base-ls960"
DEBERTA_NAME = "microsoft/deberta-v3-base"

# Existing processed utterance folders
PROCESSED_ROOT = Path(r"D:\Datasets\E-DAIC-PROCESSED")

# Separate Stage C feature output
FEATURE_ROOT = Path(r"D:\Datasets\E-DAIC-FEATURES\HUBERT_DEBERTA_SEQUENCE")

# Existing official split metadata
OFFICIAL_SPLIT_DIR = (
    Path("data") / "processed" / "official_split"
)

# New outputs
OUTPUT_ROOT = (
    Path("data") / "processed" / "quantum"
)

TRAIN_OUTPUT = OUTPUT_ROOT / "train.pkl"
DEV_OUTPUT = OUTPUT_ROOT / "dev.pkl"
TEST_OUTPUT = OUTPUT_ROOT / "test.pkl"

# Maximum DeBERTa sequence length, matching existing implementation
TEXT_MAX_LENGTH = 128

# Save pooled embeddings too, so the sequence records remain compatible
# with the existing feature representation.
SAVE_POOLED_EMBEDDINGS = True

# Device / model loading

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("CUDA device:", torch.cuda.get_device_name(0))


print("\nLoading HuBERT...")
hubert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    HUBERT_NAME
)

hubert_model = HubertModel.from_pretrained(
    HUBERT_NAME
)

hubert_model.to(DEVICE)
hubert_model.eval()


print("Loading DeBERTa...")
deberta_tokenizer = AutoTokenizer.from_pretrained(
    DEBERTA_NAME
)

deberta_model = AutoModel.from_pretrained(
    DEBERTA_NAME
)

deberta_model.to(DEVICE)
deberta_model.eval()

# Utility functions

def clean_float32_array(x: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to CPU float32 NumPy array.
    """
    return (
        x.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Match the existing Stage B audio preprocessing.

    The original pipeline reads WAV data as float32 and divides
    by 32768.0.
    """
    audio = audio.astype("float32")

    # Match the existing implementation for int16 WAV data.
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype("float32") / 32768.0

    return audio

# HuBERT sequence extraction
def extract_hubert_sequence(
    audio: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract:

        hidden_states: [T_audio, 768]
        pooled_embedding: [768]

    from HuBERT.

    The hidden sequence is the actual HuBERT last_hidden_state.
    It is NOT reconstructed from the pooled embedding.
    """

    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    inputs = hubert_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = hubert_model(
            input_values=input_values
        )

    hidden_states = outputs.last_hidden_state

    # Current Stage B behavior: mean pooling over the sequence.
    pooled = hidden_states.mean(
        dim=1
    ).squeeze(0)

    hidden_states = hidden_states.squeeze(0)

    return (
        clean_float32_array(hidden_states),
        clean_float32_array(pooled),
    )

# DeBERTa sequence extraction

def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Attention-mask-aware mean pooling.

    hidden_states:
        [batch, sequence, hidden]

    attention_mask:
        [batch, sequence]
    """
    mask = attention_mask.unsqueeze(-1).to(
        hidden_states.dtype
    )

    summed = (
        hidden_states * mask
    ).sum(dim=1)

    count = mask.sum(dim=1).clamp_min(1.0)

    return summed / count


def extract_deberta_sequence(
    text: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract:

        hidden_states: [T_text, 768]
        pooled_embedding: [768]

    from DeBERTa.

    The sequence excludes padding positions by applying the
    tokenizer attention mask.
    """

    if text is None or len(text.strip()) == 0:
        return (
            np.zeros((1, 768), dtype=np.float32),
            np.zeros(768, dtype=np.float32),
        )

    inputs = deberta_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=TEXT_MAX_LENGTH,
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = deberta_model(
            **inputs
        )

    hidden_states = outputs.last_hidden_state

    # Remove padding from the saved sequence.
    attention_mask = inputs["attention_mask"]

    valid_mask = attention_mask[0].bool()

    sequence = hidden_states[0][valid_mask]

    # Preserve the existing 768-D pooled representation,
    # but make pooling mask-aware.
    pooled = masked_mean_pool(
        hidden_states,
        attention_mask,
    )[0]

    return (
        clean_float32_array(sequence),
        clean_float32_array(pooled),
    )

# Process one participant

def process_participant(
    participant_dir: Path,
) -> List[Dict]:
    """
    Process all utterances for one participant.
    """

    samples = []

    wav_files = sorted(
        participant_dir.glob("*.wav")
    )

    print(
        f"Utterances found: {len(wav_files)}"
    )

    failed = 0

    for wav_path in tqdm(
        wav_files,
        desc=participant_dir.name,
        leave=False,
    ):
        json_path = wav_path.with_suffix(".json")

        if not json_path.exists():
            continue

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        try:
            # ------------------------------------------------
            # Audio
            # ------------------------------------------------

            sample_rate, audio = wavfile.read(
                wav_path
            )

            # Preserve the actual original dtype for
            # normalization.
            if np.issubdtype(audio.dtype, np.integer):
                audio = (
                    audio.astype("float32")
                    / 32768.0
                )
            else:
                audio = audio.astype("float32")

            (
                audio_hidden_states,
                audio_embedding,
            ) = extract_hubert_sequence(
                audio=audio,
                sample_rate=sample_rate,
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            (
                text_hidden_states,
                text_embedding,
            ) = extract_deberta_sequence(
                metadata["text"]
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if audio_hidden_states.ndim != 2:
                raise ValueError(
                    "HuBERT sequence must be 2-D"
                )

            if audio_hidden_states.shape[1] != 768:
                raise ValueError(
                    f"Unexpected HuBERT dimension: "
                    f"{audio_hidden_states.shape}"
                )

            if text_hidden_states.ndim != 2:
                raise ValueError(
                    "DeBERTa sequence must be 2-D"
                )

            if text_hidden_states.shape[1] != 768:
                raise ValueError(
                    f"Unexpected DeBERTa dimension: "
                    f"{text_hidden_states.shape}"
                )

            if not np.isfinite(
                audio_hidden_states
            ).all():
                raise ValueError(
                    "Non-finite HuBERT hidden states"
                )

            if not np.isfinite(
                text_hidden_states
            ).all():
                raise ValueError(
                    "Non-finite DeBERTa hidden states"
                )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            sample = {
                "participant_id":
                    metadata["participant_id"],

                "utterance_id":
                    metadata["utterance_id"],

                "text":
                    metadata["text"],

                "confidence":
                    metadata["confidence"],

                "duration":
                    metadata["duration"],

                # Existing pooled representations
                "audio_embedding":
                    audio_embedding,

                "text_embedding":
                    text_embedding,

                # NEW Stage C sequence representations
                "audio_hidden_states":
                    audio_hidden_states,

                "text_hidden_states":
                    text_hidden_states,
            }

            samples.append(sample)

        except Exception as e:
            failed += 1

            print(
                f"\nFailed on {wav_path.name}"
            )
            print(e)

            continue

    print(
        f"Processed: {len(samples)}"
    )

    print(
        f"Failed: {failed}"
    )

    return samples

# Load official split assignments

def load_official_participant_splits() -> Dict[str, str]:
    """
    Load the authoritative 275-participant train/dev/test
    assignments from the Stage A participant manifest.
    """

    manifest_path = (
        OFFICIAL_SPLIT_DIR / "participant_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Official participant manifest not found: "
            f"{manifest_path}"
        )

    import pandas as pd

    df = pd.read_csv(
        manifest_path,
        dtype={"participant_id": str},
    )

    required_columns = {
        "participant_id",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Participant manifest is missing columns: "
            f"{sorted(missing)}"
        )

    mapping = {}

    for _, row in df.iterrows():
        pid = str(row["participant_id"]).strip()
        split = str(row["split"]).strip()

        if split not in {"train", "dev", "test"}:
            raise ValueError(
                f"Invalid split '{split}' for participant {pid}"
            )

        mapping[pid] = split

    print(
        f"Official participant assignments: "
        f"{len(mapping)}"
    )

    print(
        f"  train: {sum(v == 'train' for v in mapping.values())}"
    )

    print(
        f"  dev:   {sum(v == 'dev' for v in mapping.values())}"
    )

    print(
        f"  test:  {sum(v == 'test' for v in mapping.values())}"
    )

    return mapping

# Main extraction

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    participant_to_split = (
        load_official_participant_splits()
    )

    participant_dirs = sorted(
        PROCESSED_ROOT.glob(
            "participant_*"
        )
    )

    print(
        f"Participant directories found: "
        f"{len(participant_dirs)}"
    )

    datasets = {
        "train": [],
        "dev": [],
        "test": [],
    }

    failed_participants = []

    for participant_dir in participant_dirs:

        participant_id = (
            participant_dir.name.split("_")[-1]
        )

        if participant_id not in participant_to_split:
            print(
                f"Skipping participant {participant_id}: "
                "not found in official split mapping."
            )
            continue

        split = participant_to_split[
            participant_id
        ]

        print("\n" + "=" * 70)
        print(
            f"Processing participant {participant_id} "
            f"→ {split}"
        )
        print("=" * 70)

        try:
            samples = process_participant(
                participant_dir
            )

            # Add split metadata
            for sample in samples:
                sample["split"] = split

            datasets[split].extend(
                samples
            )

            # Participant-level checkpoint
            participant_output = (
                FEATURE_ROOT
                / f"{participant_id}.pkl"
            )

            with open(
                participant_output,
                "wb",
            ) as f:
                pickle.dump(
                    samples,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        except Exception as e:
            failed_participants.append(
                (participant_id, str(e))
            )

    # Save split datasets

    for split, records in datasets.items():

        output_path = {
            "train": TRAIN_OUTPUT,
            "dev": DEV_OUTPUT,
            "test": TEST_OUTPUT,
        }[split]

        with open(
            output_path,
            "wb",
        ) as f:
            pickle.dump(
                records,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"\nSaved {split}: "
            f"{len(records)} records → {output_path}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE C EXTRACTION COMPLETE")
    print("=" * 70)

    for split, records in datasets.items():

        participants = sorted(
            {
                str(r["participant_id"])
                for r in records
            }
        )

        print(
            f"{split:>5}: "
            f"{len(participants):>3} participants, "
            f"{len(records):>6} utterances"
        )

    if failed_participants:
        print("\nFailed participants:")

        for pid, error in failed_participants:
            print(
                f"  {pid}: {error}"
            )

    print("\nOutputs:")
    print("  ", TRAIN_OUTPUT)
    print("  ", DEV_OUTPUT)
    print("  ", TEST_OUTPUT)


if __name__ == "__main__":
    main()
