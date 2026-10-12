import torch

import torch.nn as nn


class BaselineAttention(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=256,

        dropout=0.3,

    ):

        super().__init__()

        # ---------------------------------------------------
        # attention pooling
        # ---------------------------------------------------

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

        # ---------------------------------------------------
        # classifier
        # ---------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(

                embedding_dim,

                hidden_dim,

            ),

            nn.LayerNorm(

                hidden_dim,

            ),

            nn.GELU(),

            nn.Dropout(

                dropout,

            ),

            nn.Linear(

                hidden_dim,

                2,

            ),

        )

    # ---------------------------------------------------
    # attention pooling
    # ---------------------------------------------------

    def attention_pool(

        self,

        embeddings,

        attention_mask,

    ):

        # (B,L,1)

        scores = self.attention(

            embeddings

        )

        # ignore padded positions

        scores = scores.masked_fill(

            attention_mask.unsqueeze(-1) == 0,

            -1e9,

        )

        # (B,L,1)

        weights = torch.softmax(

            scores,

            dim=1,

        )

        pooled = torch.sum(

            weights * embeddings,

            dim=1,

        )

        return pooled

    # ---------------------------------------------------
    # forward
    # ---------------------------------------------------

    def forward(

        self,

        embeddings,

        attention_mask,

    ):

        pooled = self.attention_pool(

            embeddings,

            attention_mask,

        )

        logits = self.classifier(

            pooled,

        )

        return logits