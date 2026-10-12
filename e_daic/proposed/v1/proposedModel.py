import torch
import torch.nn as nn

from src.proposed.state_space import (
    StateProjection,
)

from src.proposed.discordance import (
    cosine_discordance,
    euclidean_discordance,
    entropy_discordance,
)

from src.proposed.kraus import (
    KrausEvolution,
)


class ProposedDiscordanceModel(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        latent_dim=128,
    ):
        super().__init__()

        self.state_projection = (
            StateProjection(
                embedding_dim,
                latent_dim,
            )
        )

        self.kraus = KrausEvolution(
            latent_dim
        )

        fusion_dim = (
            latent_dim * 2 + 3
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2,
            ),
        )

    def forward(
        self,
        audio_embedding,
        text_embedding,
    ):

        # ---------------------------------------------------
        # latent states
        # ---------------------------------------------------

        (
            audio_state,
            text_state,
        ) = self.state_projection(
            audio_embedding,
            text_embedding,
        )

        # ---------------------------------------------------
        # kraus-inspired evolution
        # ---------------------------------------------------

        audio_state = self.kraus(
            audio_state
        )

        text_state = self.kraus(
            text_state
        )

        # ---------------------------------------------------
        # discordance metrics
        # ---------------------------------------------------

        cosine_disc = (
            cosine_discordance(
                audio_state,
                text_state,
            ).unsqueeze(-1)
        )

        euclidean_disc = (
            euclidean_discordance(
                audio_state,
                text_state,
            ).unsqueeze(-1)
        )

        entropy_disc = (
            entropy_discordance(
                audio_state,
                text_state,
            ).unsqueeze(-1)
        )

        # ---------------------------------------------------
        # fusion
        # ---------------------------------------------------

        fused = torch.cat([

            audio_state,
            text_state,

            cosine_disc,
            euclidean_disc,
            entropy_disc,

        ], dim=-1)

        return self.classifier(
            fused
        )