import torch
import torch.nn as nn


class KrausInteraction(nn.Module):

    def __init__(
        self,
        latent_dim=128,
    ):
        super().__init__()

        self.cross_modal_gate = nn.Sequential(

            nn.Linear(
                latent_dim * 2,
                latent_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                latent_dim,
                latent_dim,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_state,
        text_state,
    ):

        joint = torch.cat([

            audio_state,
            text_state,

        ], dim=-1)

        interaction = (
            self.cross_modal_gate(
                joint
            )
        )

        evolved_audio = (
            audio_state *
            interaction
        )

        evolved_text = (
            text_state *
            interaction
        )

        return (
            evolved_audio,
            evolved_text,
        )