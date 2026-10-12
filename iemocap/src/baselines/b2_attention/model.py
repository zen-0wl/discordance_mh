"""
HuBERT + DeBERTa

Cross-modal attention
followed by an MLP classifier.
"""

import torch
import torch.nn as nn

from src.config.emotion_config import NUM_CLASSES


class AttentionFusion(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=512,

        num_heads=8,

        dropout=0.30,

        num_classes=NUM_CLASSES,

    ):

        super().__init__()

        self.audio_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.text_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.cross_attention = nn.MultiheadAttention(

            embed_dim=hidden_dim,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True,

        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

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

        # project into common space

        audio = self.audio_projection(
            audio
        )

        text = self.text_projection(
            text
        )

        # (B,D) -> (B,1,D)

        audio = audio.unsqueeze(1)

        text = text.unsqueeze(1)

        # text attends to audio

        attended, _ = self.cross_attention(

            query=text,

            key=audio,

            value=audio,

        )

        attended = attended.squeeze(1)

        logits = self.classifier(
            attended
        )

        return logits