import torch

import torch.nn as nn

import torch.nn.functional as F


# ---------------------------------------------------
# Quantum Head
# ---------------------------------------------------

class QuantumHead(nn.Module):

    def __init__(

        self,

        hidden_dim=256,

    ):

        super().__init__()

        # amplitude projection

        self.amplitude = nn.Linear(

            hidden_dim,

            hidden_dim,

        )

        # learnable phase rotation

        self.phase = nn.Parameter(

            torch.randn(hidden_dim)

        )

        self.layer_norm = nn.LayerNorm(

            hidden_dim

        )

    def forward(

        self,

        x,

    ):

        # --------------------------------------------
        # amplitude encoding
        # --------------------------------------------

        amplitude = self.amplitude(

            x

        )

        amplitude = F.normalize(

            amplitude,

            p=2,

            dim=-1,

        )

        # --------------------------------------------
        # phase rotation
        # --------------------------------------------

        phase = torch.cos(

            self.phase

        )

        quantum = amplitude * phase

        # --------------------------------------------
        # residual connection
        # --------------------------------------------

        quantum = quantum + x

        quantum = self.layer_norm(

            quantum

        )

        return quantum


# ---------------------------------------------------
# Baseline Quantum
# ---------------------------------------------------

class BaselineQuantum(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=256,

        dropout=0.3,

    ):

        super().__init__()

        # --------------------------------------------
        # attention pooling
        # --------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(

                embedding_dim,

                hidden_dim,

            ),

            nn.Tanh(),

            nn.Linear(

                hidden_dim,

                1,

            ),

        )

        # --------------------------------------------
        # projection
        # --------------------------------------------

        self.projection = nn.Sequential(

            nn.Linear(

                embedding_dim,

                hidden_dim,

            ),

            nn.LayerNorm(

                hidden_dim,

            ),

            nn.GELU(),

        )

        # --------------------------------------------
        # quantum head
        # --------------------------------------------

        self.quantum = QuantumHead(

            hidden_dim

        )

        # --------------------------------------------
        # classifier
        # --------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(

                dropout,

            ),

            nn.Linear(

                hidden_dim,

                2,

            ),

        )

    # ------------------------------------------------
    # attention pooling
    # ------------------------------------------------

    def attention_pool(

        self,

        embeddings,

        attention_mask,

    ):

        scores = self.attention(

            embeddings

        )

        scores = scores.masked_fill(

            attention_mask.unsqueeze(-1) == 0,

            -1e9,

        )

        weights = torch.softmax(

            scores,

            dim=1,

        )

        pooled = torch.sum(

            weights * embeddings,

            dim=1,

        )

        return pooled

    # ------------------------------------------------
    # forward
    # ------------------------------------------------

    def forward(

        self,

        embeddings,

        attention_mask,

    ):

        pooled = self.attention_pool(

            embeddings,

            attention_mask,

        )

        projected = self.projection(

            pooled,

        )

        quantum = self.quantum(

            projected,

        )

        logits = self.classifier(

            quantum,

        )

        return logits