import torch

import torch.nn as nn


class BaselineTransformer(nn.Module):

    def __init__(

        self,

        embedding_dim=768,

        hidden_dim=256,

        dropout=0.3,

    ):

        super().__init__()

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
    # masked mean pooling
    # ---------------------------------------------------

    def mean_pool(

        self,

        embeddings,

        attention_mask,

    ):

        mask = attention_mask.unsqueeze(-1).float()

        summed = torch.sum(

            embeddings * mask,

            dim=1,

        )

        counts = torch.clamp(

            mask.sum(dim=1),

            min=1e-9,

        )

        pooled = summed / counts

        return pooled

    # ---------------------------------------------------
    # forward
    # ---------------------------------------------------

    def forward(

        self,

        embeddings,

        attention_mask,

    ):

        pooled = self.mean_pool(

            embeddings,

            attention_mask,

        )

        logits = self.classifier(

            pooled,

        )

        return logits