import torch.nn as nn
import torch.nn.functional as F


class StateProjection(nn.Module):

    def __init__(
        self,
        input_dim=768,
        latent_dim=128,
    ):
        super().__init__()

        self.audio_proj = nn.Linear(
            input_dim,
            latent_dim,
        )

        self.text_proj = nn.Linear(
            input_dim,
            latent_dim,
        )

    def forward(
        self,
        audio_embedding,
        text_embedding,
    ):

        audio_state = F.normalize(

            self.audio_proj(
                audio_embedding
            ),

            dim=-1,
        )

        text_state = F.normalize(

            self.text_proj(
                text_embedding
            ),

            dim=-1,
        )

        return (
            audio_state,
            text_state,
        )