import torch
import torch.nn as nn


class KrausEvolution(nn.Module):

    def __init__(
        self,
        latent_dim=128,
    ):
        super().__init__()

        self.noise = nn.Linear(
            latent_dim,
            latent_dim,
        )

    def forward(
        self,
        state,
    ):

        evolved = (
            state +
            0.1 * torch.tanh(
                self.noise(state)
            )
        )

        return evolved