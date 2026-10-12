"""
Transformer Embeddings
(HuBERT + DeBERTa)

Mean pooled embeddings
followed by a simple MLP.
"""

import torch
import torch.nn as nn

from src.config.emotion_config import NUM_CLASSES


class TransformerFusion(nn.Module):

    def __init__(

        self,

        audio_dim=768,

        text_dim=768,

        hidden_dim=512,

        dropout=0.30,

        num_classes=NUM_CLASSES,

    ):

        super().__init__()

        fusion_dim = (

            audio_dim +

            text_dim

        )

        self.classifier = nn.Sequential(

            nn.Linear(

                fusion_dim,

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

        fused = torch.cat(

            [

                audio,

                text,

            ],

            dim=1,

        )

        logits = self.classifier(

            fused,

        )

        return logits