from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
)


class TextFeatureExtractor:
    """
    DeBERTa-v3 feature extractor.

    Produces one 768-dimensional embedding
    for each transcript.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device: str | None = None,
        max_length: int = 128,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.max_length = max_length

        print(f"Loading DeBERTa on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        self.model = (
            AutoModel
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

    # ---------------------------------------------------------

    @staticmethod
    def mean_pooling(
        last_hidden_state,
        attention_mask,
    ):
        """
        Mean pooling excluding padding tokens.
        """

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        summed = torch.sum(
            last_hidden_state * mask,
            dim=1,
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

        return summed / counts

    # ---------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        text: str,
    ) -> np.ndarray:

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokens = {
            k: v.to(self.device)
            for k, v in tokens.items()
        }

        outputs = self.model(**tokens)

        embedding = self.mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )

        embedding = (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embedding

    # ---------------------------------------------------------

    @torch.no_grad()
    def batch_extract(
        self,
        texts,
    ):

        tokens = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokens = {
            k: v.to(self.device)
            for k, v in tokens.items()
        }

        outputs = self.model(**tokens)

        embeddings = self.mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embeddings

    # ---------------------------------------------------------

    def extract_dataset(
        self,
        texts,
        batch_size: int = 32,
    ):

        embeddings = []

        total = len(texts)

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = start + batch_size

            batch = texts[start:end]

            batch_embeddings = (
                self.batch_extract(batch)
            )

            embeddings.extend(batch_embeddings)

        return embeddings

    # ---------------------------------------------------------

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