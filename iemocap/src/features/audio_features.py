from pathlib import Path

import librosa
import numpy as np
import torch

from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
)


class AudioFeatureExtractor:
    """
    HuBERT speech feature extractor.

    Produces one 768-dimensional embedding
    for each utterance using mean pooling.
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str | None = None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(f"Loading HuBERT model on {self.device}...")

        self.processor = (
            Wav2Vec2FeatureExtractor
            .from_pretrained(model_name)
        )

        self.model = (
            HubertModel
            .from_pretrained(model_name)
            .to(self.device)
        )

        self.model.eval()

        self.embedding_dim = (
            self.model.config.hidden_size
        )

        print(
            f"Embedding dimension: {self.embedding_dim}"
        )

    # -----------------------------------------------------

    def load_audio(
        self,
        wav_path: str | Path,
        sample_rate: int = 16000,
    ):

        speech, _ = librosa.load(
            wav_path,
            sr=sample_rate,
            mono=True,
        )

        return speech

    # -----------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        wav_path: str | Path,
    ) -> np.ndarray:

        speech = self.load_audio(wav_path)

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        embedding = (
            outputs.last_hidden_state
            .mean(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        return embedding.astype(np.float32)

    # -----------------------------------------------------

    @torch.no_grad()
    def batch_extract(
        self,
        wav_paths,
    ):

        speeches = [
            self.load_audio(path)
            for path in wav_paths
        ]

        inputs = self.processor(
            speeches,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        embeddings = (
            outputs.last_hidden_state
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embeddings

    # -----------------------------------------------------

    def extract_dataset(
        self,
        wav_paths,
        batch_size=16,
    ):

        embeddings = []

        total = len(wav_paths)

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = start + batch_size

            batch = wav_paths[start:end]

            batch_embeddings = (
                self.batch_extract(batch)
            )

            embeddings.extend(batch_embeddings)

        return embeddings

    # -----------------------------------------------------

    @staticmethod
    def validate_embedding(
        embedding: np.ndarray,
    ):

        if embedding.shape != (768,):
            raise ValueError(
                f"Unexpected embedding shape: {embedding.shape}"
            )

        if np.isnan(embedding).any():
            raise ValueError(
                "Embedding contains NaN values."
            )

        if np.isinf(embedding).any():
            raise ValueError(
                "Embedding contains Inf values."
            )

        if np.allclose(embedding, 0):
            raise ValueError(
                "Embedding is entirely zeros."
            )

        return True