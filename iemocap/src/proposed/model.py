import torch
import torch.nn as nn

from src.config.emotion_config import NUM_CLASSES

from .density import DensityRepresentation
from .quantum_metrics import discordance_vector
from .reliability import (
    ReliabilityGate,
    ReliabilityFusion,
)


class FeatureGate(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                embedding_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),

            nn.Sigmoid(),

        )

    def forward(
        self,
        audio,
        text,
    ):

        gate = self.network(

            torch.cat(
                [
                    audio,
                    text,
                ],
                dim=1,
            )

        )

        audio = gate * audio

        text = (1.0 - gate) * text

        return audio, text


class DiscordanceNet(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=512,

        dropout=0.30,

        num_classes=NUM_CLASSES,

    ):

        super().__init__()

        self.feature_gate = FeatureGate(
            embedding_dim,
        )

        self.audio_density = DensityRepresentation(

            embedding_dim,

            hidden_dim,

        )

        self.text_density = DensityRepresentation(

            embedding_dim,

            hidden_dim,

        )

        self.reliability_gate = ReliabilityGate(

            metric_dim=4,

            hidden_dim=hidden_dim,

        )

        self.reliability_fusion = ReliabilityFusion()

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim * 2 + 4,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim // 2,
                num_classes,
            ),

        )

    # --------------------------------------------------

    def forward(
        self,
        audio,
        text,
    ):

        # -------------------------------
        # Feature Gate
        # -------------------------------

        audio, text = self.feature_gate(

            audio,

            text,

        )

        # -------------------------------
        # Quantum States
        # -------------------------------

        psi_audio, rho_audio = self.audio_density(

            audio,

        )

        psi_text, rho_text = self.text_density(

            text,

        )

        # -------------------------------
        # Discordance Metrics
        # -------------------------------

        discordance = discordance_vector(

            rho_audio,

            rho_text,

        )

        # -------------------------------
        # Reliability
        # -------------------------------

        reliability = self.reliability_gate(

            discordance,

        )

        # -------------------------------
        # Feature Fusion
        # -------------------------------

        audio_features = torch.abs(psi_audio)

        text_features = torch.abs(psi_text)

        difference_features = torch.abs(
            psi_audio - psi_text
        )

        # Preserve the original feature size while explicitly modelling disagreement.

        fused = (
            audio_features
            + text_features
            + difference_features
        ) / 3.0

        fused = self.reliability_fusion(

            fused,

            1.0 + reliability,

        )

        # -------------------------------
        # Classification
        # -------------------------------

        classifier_input = torch.cat(
            [
                fused,
                reliability,
                discordance,
            ],
            dim=1,
        )

        logits = self.classifier(classifier_input)

        return logits