"""
Reliability-aware discordance weighting.

Uses the quantum discordance metrics to estimate
how reliable the multimodal representation is.
"""

import torch
import torch.nn as nn


class ReliabilityGate(nn.Module):
    """
    Learns feature-wise reliability weights from
    the discordance vector.

    Input:
        discordance : (B, 4)

    Output:
        reliability : (B, hidden_dim)
    """

    def __init__(

        self,

        metric_dim=4,

        hidden_dim=512,

        dropout=0.20,

    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                metric_dim,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                64,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                128,
                hidden_dim,
            ),

            nn.Sigmoid(),

        )

    def forward(
        self,
        discordance,
    ):

        return self.network(
            discordance,
        )


class ReliabilityFusion(nn.Module):
    """
    Applies reliability weights to quantum features.

    Output

        gated = r * features
    """

    def forward(

        self,

        features,

        reliability,

    ):

        return (

            features

            * reliability

        )