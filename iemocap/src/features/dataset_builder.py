"""

Pipeline

   metadata.csv
        ↓
     HuBERT
        ↓
     DeBERTa
        ↓
   dataset.pkl
"""
from pathlib import Path
import pickle
import torch

import pandas as pd
from tqdm.auto import tqdm

from src.features.audio_features import AudioFeatureExtractor
from src.features.text_features import TextFeatureExtractor

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

METADATA_FILE = Path(
    r"D:\Datasets\IEMOCAP_withoutVid\metadata.csv"
)

class FeatureBuilder:

    def __init__(
        self,
        metadata_path,
        audio_batch_size=16,
        text_batch_size=32,
    ):

        self.metadata_path = Path(metadata_path)

        self.output_dir = PROCESSED_DIR

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.audio_batch_size = audio_batch_size
        self.text_batch_size = text_batch_size

        self.audio_extractor = AudioFeatureExtractor()

        self.text_extractor = TextFeatureExtractor()

    # --------------------------------------------------

    def load_metadata(self):

        print("Loading metadata...")

        metadata = pd.read_csv(
            self.metadata_path
        )

        print(
            f"{len(metadata)} utterances loaded."
        )

        return metadata

    # --------------------------------------------------

    def build_dataset(self):

        metadata = self.load_metadata()

        dataset = []

        total = len(metadata)

        batch_size = min(
            self.audio_batch_size,
            self.text_batch_size,
        )

        num_batches = (
            total + batch_size - 1
        ) // batch_size

        print()
        print("Building multimodal dataset...")

        progress = tqdm(
            range(0, total, batch_size),
            total=num_batches,
            desc="Building Dataset",
            unit="batch",
            dynamic_ncols=True,
        )

        for start in progress:

            end = min(start + batch_size, total)

            batch = metadata.iloc[start:end]

            wav_paths = batch["wav_path"].tolist()

            transcripts = batch["transcript"].tolist()

            audio_embeddings = self.audio_extractor.batch_extract(
                wav_paths
            )

            text_embeddings = self.text_extractor.batch_extract(
                transcripts
            )
            
            torch.cuda.empty_cache()

            for i, (_, row) in enumerate(batch.iterrows()):

                sample = {

                    "utterance_id": row["utterance_id"],
                    "session": row["session"],
                    "dialogue": row["dialogue"],
                    "speaker": row["speaker"],
                    "emotion": row["emotion"],
                    "transcript": row["transcript"],
                    "start": row["start"],
                    "end": row["end"],
                    "duration": row["duration"],

                    "audio_embedding": audio_embeddings[i],
                    "text_embedding": text_embeddings[i],

                }

                dataset.append(sample)

            progress.set_postfix(
                samples=len(dataset)
            )

        print()

        print(
            f"Dataset built with {len(dataset)} samples."
        )

        return dataset

    # --------------------------------------------------

    def save_dataset(
        self,
        dataset,
        filename="dataset.pkl",
    ):

        output = (
            self.output_dir
            / filename
        )

        with open(
            output,
            "wb",
        ) as f:

            pickle.dump(
                dataset,
                f,
            )

        print()

        print(
            f"Dataset saved to:\n{output}"
        )

        return output

    # --------------------------------------------------

    @staticmethod
    def load_dataset(path):

        with open(
            path,
            "rb",
        ) as f:

            dataset = pickle.load(f)

        return dataset

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    from src.features.validation import (
        validate_dataset,
        dataset_summary,
    )

    builder = FeatureBuilder(

        metadata_path=METADATA_FILE,

        audio_batch_size=16,

        text_batch_size=32,

    )

    dataset = builder.build_dataset()

    validate_dataset(dataset)

    dataset_summary(dataset)

    builder.save_dataset(dataset)

    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()