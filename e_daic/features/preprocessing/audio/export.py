from pathlib import Path
import json

import soundfile as sf

from src.dataset.schema import UtteranceSample


def export_utterance(
    utterance: UtteranceSample,
    output_dir: Path,
):
    """
    Export single utterance audio + metadata.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # audio path
    audio_path = output_dir / f"{utterance.utterance_id}.wav"

    # metadata path
    metadata_path = output_dir / f"{utterance.utterance_id}.json"

    # save audio
    sf.write(
        audio_path,
        utterance.audio,
        utterance.sample_rate,
    )

    # metadata
    metadata = {
        "participant_id": utterance.participant_id,
        "utterance_id": utterance.utterance_id,
        "start_time": utterance.start_time,
        "end_time": utterance.end_time,
        "duration": utterance.duration,
        "text": utterance.text,
        "confidence": utterance.confidence,
        "sample_rate": utterance.sample_rate,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def export_utterances(
    utterances,
    participant_output_dir: Path,
):
    """
    Export list of utterances.
    """

    for utterance in utterances:
        export_utterance(
            utterance,
            participant_output_dir,
        )