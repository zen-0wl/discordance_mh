"""
Quantum Head Baseline

HuBERT + DeBERTa

Feature-level gating
followed by a lightweight
quantum-inspired complex-valued head.
"""

import torch
import torch.nn as nn

from src.config.emotion_config import NUM_CLASSES


class FeatureGate(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
    ):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                embedding_dim,
            ),

            nn.Sigmoid(),

        )

    def forward(
        self,
        audio,
        text,
    ):

        fused = torch.cat(
            [
                audio,
                text,
            ],
            dim=1,
        )

        gate = self.gate(
            fused
        )

        gated_audio = gate * audio

        gated_text = (
            1.0 - gate
        ) * text

        fused = torch.cat(
            [
                gated_audio,
                gated_text,
            ],
            dim=1,
        )

        return fused


class QuantumHead(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=512,
    ):

        super().__init__()

        self.real_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.imag_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.phase = nn.Parameter(
            torch.zeros(hidden_dim)
        )

    def forward(
        self,
        x,
    ):

        real = self.real_projection(
            x
        )

        imag = self.imag_projection(
            x
        )

        complex_state = torch.complex(
            real,
            imag,
        )

        phase = torch.exp(
            1j * self.phase
        )

        complex_state = (
            complex_state * phase
        )

        magnitude = torch.abs(
            complex_state
        )

        return magnitude


# ======================================================
# Model
# ======================================================

class QuantumGatedFusion(nn.Module):

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

        self.quantum_head = QuantumHead(

            input_dim=embedding_dim * 2,

            hidden_dim=hidden_dim,

        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
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

        fused = self.feature_gate(

            audio,

            text,

        )

        quantum_features = self.quantum_head(

            fused,

        )

        logits = self.classifier(

            quantum_features,

        )

        return logits