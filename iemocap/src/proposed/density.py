"""
Quantum-inspired density state construction.

Converts real-valued multimodal features into
complex-valued quantum states and density matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumStateEncoder(nn.Module):
    """
    Learn a complex-valued latent representation.

    ψ = real + i * imag
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
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

        # Learnable global phase
        self.phase = nn.Parameter(
            torch.zeros(hidden_dim)
        )

    def forward(
        self,
        x,
    ):

        real = self.real_projection(x)

        imag = self.imag_projection(x)

        psi = torch.complex(
            real,
            imag,
        )

        phase = torch.exp(
            1j * self.phase
        )

        psi = psi * phase

        # L2 normalization

        norm = torch.linalg.norm(
            psi,
            dim=-1,
            keepdim=True,
        )

        psi = psi / (
            norm + 1e-8
        )

        return psi


class DensityMatrix(nn.Module):
    """
    Construct

        ρ = |ψ><ψ|

    guaranteeing

    - Hermitian
    - Positive semi-definite
    - Trace = 1
    """

    def forward(
        self,
        psi,
    ):

        rho = torch.einsum(

            "bi,bj->bij",

            psi,

            torch.conj(
                psi,
            ),

        )

        trace = torch.diagonal(

            rho,

            dim1=-2,

            dim2=-1,

        ).sum(
            dim=-1,
            keepdim=True,
        )

        trace = trace.unsqueeze(-1)

        rho = rho / (
            trace + 1e-8
        )

        return rho


class DensityRepresentation(nn.Module):

    """
    Convenience wrapper.

    Input

        features

    Output

        ψ
        ρ
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
    ):

        super().__init__()

        self.encoder = QuantumStateEncoder(

            input_dim,

            hidden_dim,

        )

        self.density = DensityMatrix()

    def forward(
        self,
        x,
    ):

        psi = self.encoder(
            x,
        )

        rho = self.density(
            psi,
        )

        return psi, rho