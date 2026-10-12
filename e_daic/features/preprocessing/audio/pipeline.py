from pathlib import Path

import pandas as pd

from src.preprocessing.audio.segment import segment_audio
from src.preprocessing.audio.export import export_utterances


def process_participant(
    participant_dir: Path,
    output_root: Path,
):
    """
    Process a single E-DAIC participant.
    """

    participant_id = participant_dir.name.split("_")[0]

    print(f"\nProcessing participant {participant_id}")

    # ---------------------------------------------------
    # paths
    # ---------------------------------------------------

    inner_dir = participant_dir / participant_dir.name

    audio_path = inner_dir / f"{participant_id}_AUDIO.wav"

    transcript_path = inner_dir / f"{participant_id}_Transcript.csv"

    # ---------------------------------------------------
    # validate existence
    # ---------------------------------------------------

    if not audio_path.exists():
        print(f"Missing audio: {audio_path}")
        return None

    if not transcript_path.exists():
        print(f"Missing transcript: {transcript_path}")
        return None

    # ---------------------------------------------------
    # load transcript
    # ---------------------------------------------------

    transcript_df = pd.read_csv(transcript_path)

    total_rows = len(transcript_df)

    print(f"Transcript rows: {total_rows}")

    # ---------------------------------------------------
    # segment utterances
    # ---------------------------------------------------

    utterances = segment_audio(
        audio_path=str(audio_path),
        transcript_df=transcript_df,
        participant_id=participant_id,
    )

    valid_utterances = len(utterances)

    filtered_out = total_rows - valid_utterances

    print(f"Valid utterances: {valid_utterances}")
    print(f"Filtered utterances: {filtered_out}")

    # ---------------------------------------------------
    # export
    # ---------------------------------------------------

    participant_output_dir = (
        output_root / f"participant_{participant_id}"
    )

    export_utterances(
        utterances,
        participant_output_dir,
    )

    print(f"Exported to: {participant_output_dir}")

    # ---------------------------------------------------
    # return summary
    # ---------------------------------------------------

    return {
        "participant_id": participant_id,
        "total_rows": total_rows,
        "valid_utterances": valid_utterances,
        "filtered_out": filtered_out,
    }

def process_dataset(
    dataset_root: Path,
    output_root: Path,
):
    """
    Process all participant folders.
    """

    participants = sorted(
        dataset_root.glob("*_P")
    )

    print(f"Participants found: {len(participants)}")

    summaries = []

    total_valid = 0
    total_filtered = 0

    for participant_dir in participants:

        summary = process_participant(
            participant_dir,
            output_root,
        )

        if summary is not None:

            summaries.append(summary)

            total_valid += summary["valid_utterances"]
            total_filtered += summary["filtered_out"]

    # ---------------------------------------------------
    # dataset summary
    # ---------------------------------------------------

    print("\n===================================")
    print("DATASET PREPROCESSING SUMMARY")
    print("===================================")

    print(f"Participants processed: {len(summaries)}")
    print(f"Total valid utterances: {total_valid}")
    print(f"Total filtered utterances: {total_filtered}")

    print("\nDataset preprocessing complete.")

    return summaries

if __name__ == "__main__":

    dataset_root = Path(r"D:\Datasets\E-DAIC")

    output_root = Path(r"D:\Datasets\E-DAIC-PROCESSED")

    process_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
    )