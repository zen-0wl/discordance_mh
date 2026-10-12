import torch
import torch.nn as nn
import torch.nn.functional as F


class StateProjection(nn.Module):

    def __init__(
        self,
        input_dim=768,
        latent_dim=128,
        num_samples=8,
    ):
        super().__init__()

        self.num_samples = num_samples

        self.audio_proj = nn.Linear(
            input_dim,
            latent_dim,
        )

        self.text_proj = nn.Linear(
            input_dim,
            latent_dim,
        )

        self.dropout = nn.Dropout(0.15)

    def build_sequence(
        self,
        x,
        projector,
    ):

        samples = []

        for _ in range(self.num_samples):

            noisy = self.dropout(x)

            projected = projector(noisy)

            projected = F.normalize(
                projected,
                dim=-1,
            )

            samples.append(projected)

        stacked = torch.stack(
            samples,
            dim=1,
        )

        return stacked

    def forward(
        self,
        audio_embedding,
        text_embedding,
    ):

        audio_states = self.build_sequence(
            audio_embedding,
            self.audio_proj,
        )

        text_states = self.build_sequence(
            text_embedding,
            self.text_proj,
        )

        return audio_states, text_states