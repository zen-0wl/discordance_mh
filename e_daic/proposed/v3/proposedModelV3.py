import torch
import torch.nn as nn

from src.proposed.state_space import (
    StateProjection,
)

from src.proposed.density import (
    covariance_density_matrix,
)

from src.proposed.quantum_metrics import (
    trace_distance,
    von_neumann_entropy,
    fidelity,
)

from src.proposed.krausV3 import (
    KrausInteraction,
)


class ProposedDiscordanceModelV3(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        latent_dim=128,
    ):
        super().__init__()

        # ---------------------------------------------------
        # latent state projection
        # ---------------------------------------------------

        self.state_projection = (
            StateProjection(
                embedding_dim,
                latent_dim,
            )
        )

        # ---------------------------------------------------
        # kraus interaction
        # ---------------------------------------------------

        self.kraus = KrausInteraction(
            latent_dim
        )

        # ---------------------------------------------------
        # discordance weighting
        # ---------------------------------------------------

        self.metric_gate = nn.Sequential(

            nn.Linear(4, 16),

            nn.ReLU(),

            nn.Linear(16, 4),

            nn.Sigmoid(),
        )

        # ---------------------------------------------------
        # classifier
        # ---------------------------------------------------

        fusion_dim = (
            latent_dim * 2 + 4
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
        # state projection
        # ---------------------------------------------------

        (
            audio_state,
            text_state,
        ) = self.state_projection(
            audio_embedding,
            text_embedding,
        )

        # ---------------------------------------------------
        # kraus interaction
        # ---------------------------------------------------

        (
            audio_state,
            text_state,
        ) = self.kraus(
            audio_state,
            text_state,
        )

        # ---------------------------------------------------
        # density states
        # ---------------------------------------------------

        rho_audio = covariance_density_matrix(
            audio_state
        )

        rho_text = covariance_density_matrix(
            text_state
        )

        # ---------------------------------------------------
        # quantum discordance metrics
        # ---------------------------------------------------

        td = trace_distance(
            rho_audio,
            rho_text,
        ).unsqueeze(-1)

        entropy_audio = (
            von_neumann_entropy(
                rho_audio
            ).unsqueeze(-1)
        )

        entropy_text = (
            von_neumann_entropy(
                rho_text
            ).unsqueeze(-1)
        )

        fidelity_score = (
            fidelity(
                rho_audio,
                rho_text,
            ).unsqueeze(-1)
        )

        entropy_gap = torch.abs(
            entropy_audio -
            entropy_text
        )

        metrics = torch.cat([

            td,
            entropy_gap,
            fidelity_score,
            td * entropy_gap,

        ], dim=-1)

        # ---------------------------------------------------
        # adaptive weighting
        # ---------------------------------------------------

        weights = self.metric_gate(
            metrics
        )

        weighted_metrics = (
            metrics * weights
        )
        
        audio_state = audio_state.mean(dim=1)
        text_state = text_state.mean(dim=1)
        # ---------------------------------------------------
        # fusion
        # ---------------------------------------------------

        fused = torch.cat([

            audio_state,
            text_state,
            weighted_metrics,

        ], dim=-1)

        return self.classifier(
            fused
        )