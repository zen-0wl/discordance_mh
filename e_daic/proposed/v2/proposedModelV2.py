import torch
import torch.nn as nn
import torch.nn.functional as F

from src.proposed.state_spacev2 import (
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


class ProposedDiscordanceModelV2(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        latent_dim=128,
    ):
        super().__init__()

        # ---------------------------------------------------
        # shared latent state space
        # ---------------------------------------------------

        self.state_projection = (
            StateProjection(
                embedding_dim,
                latent_dim,
            )
        )

        # ---------------------------------------------------
        # cross-modal attention
        # ---------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True,
        )

        # ---------------------------------------------------
        # adaptive kraus evolution
        # ---------------------------------------------------

        self.kraus = KrausEvolution(
            latent_dim
        )

        # ---------------------------------------------------
        # learnable discordance weighting
        # ---------------------------------------------------

        self.discordance_gate = nn.Sequential(

            nn.Linear(3, 16),

            nn.ReLU(),

            nn.Linear(16, 3),

            nn.Sigmoid(),
        )

        # ---------------------------------------------------
        # classifier
        # ---------------------------------------------------

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
        # latent state projection
        # ---------------------------------------------------

        (
            audio_state,
            text_state,
        ) = self.state_projection(
            audio_embedding,
            text_embedding,
        )

        # ---------------------------------------------------
        # cross-modal attention
        # ---------------------------------------------------

        audio_seq = audio_state.unsqueeze(1)

        text_seq = text_state.unsqueeze(1)

        attended_audio, _ = (
            self.cross_attention(
                audio_seq,
                text_seq,
                text_seq,
            )
        )

        attended_text, _ = (
            self.cross_attention(
                text_seq,
                audio_seq,
                audio_seq,
            )
        )

        audio_state = attended_audio.squeeze(1)

        text_state = attended_text.squeeze(1)

        # ---------------------------------------------------
        # adaptive kraus evolution
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

        discordance = torch.cat([

            cosine_disc,
            euclidean_disc,
            entropy_disc,

        ], dim=-1)

        # ---------------------------------------------------
        # learnable weighting
        # ---------------------------------------------------

        weights = self.discordance_gate(
            discordance
        )

        weighted_discordance = (
            discordance * weights
        )

        # ---------------------------------------------------
        # final fusion
        # ---------------------------------------------------

        fused = torch.cat([

            audio_state,
            text_state,
            weighted_discordance,

        ], dim=-1)

        return self.classifier(
            fused
        )