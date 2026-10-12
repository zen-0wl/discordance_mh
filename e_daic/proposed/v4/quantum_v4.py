from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-8

# Basic utilities

def symmetrize_density(rho: torch.Tensor) -> torch.Tensor:
    """
    Enforce numerical symmetry/Hermiticity for real-valued
    density matrices.
    """
    return 0.5 * (rho + rho.transpose(-1, -2))


def normalize_state_vectors(
    x: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Normalize vectors along the feature dimension.

    Input:
        x: [..., T, D]

    Output:
        normalized vectors with unit L2 norm.
    """
    norms = torch.linalg.vector_norm(
        x,
        dim=-1,
        keepdim=True,
    )

    return x / norms.clamp_min(eps)

# Mixed density-state construction

def build_mixed_density(
    sequence: torch.Tensor,
    attention_logits: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Construct an attention-weighted mixed density state.

    Implements:

        rho = sum_k alpha_k psi_k psi_k^T

    where:

        psi_k = z_k / ||z_k||

        alpha = softmax(attention_logits)

    Parameters
    ----------
    sequence:
        [B, T, R]

    attention_logits:
        [B, T]

    mask:
        [B, T]
        True for valid positions, False for padding.

    Returns
    -------
    rho:
        [B, R, R]

    Notes
    -----
    The convex mixture of normalized outer products produces
    a positive semidefinite unit-trace density matrix.
    """

    # Normalize each contextualized token/frame vector.
    states = normalize_state_vectors(sequence, eps)

    # Mask padded positions before softmax.
    masked_logits = attention_logits.masked_fill(
        ~mask,
        -1e9,
    )

    alpha = torch.softmax(
        masked_logits,
        dim=-1,
    )

    # Safety: zero padded positions explicitly.
    alpha = alpha * mask.to(alpha.dtype)

    alpha = alpha / alpha.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)

    # Outer products:
    #
    # psi_k psi_k^T
    #
    # [B,T,R] -> [B,T,R,R]

    outer = states.unsqueeze(-1) * states.unsqueeze(-2)

    # Weighted mixture.
    rho = torch.sum(
        alpha.unsqueeze(-1).unsqueeze(-1) * outer,
        dim=1,
    )

    rho = symmetrize_density(rho)

    # Numerical trace normalization.
    trace = torch.diagonal(
        rho,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1, keepdim=True)

    rho = rho / trace.unsqueeze(-1).clamp_min(eps)

    return rho

# Quantum metrics

def density_eigenvalues(
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    Return eigenvalues of a symmetric density matrix.
    """
    rho = symmetrize_density(rho)

    eigenvalues = torch.linalg.eigvalsh(rho)

    # Remove tiny negative values caused by floating-point
    # numerical error.
    return torch.clamp(
        eigenvalues,
        min=0.0,
    )


def von_neumann_entropy(
    rho: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    S(rho) = -Tr(rho log rho)
    """

    eigenvalues = density_eigenvalues(rho)

    safe = eigenvalues.clamp_min(eps)

    entropy = -torch.sum(
        eigenvalues * torch.log(safe),
        dim=-1,
    )

    return entropy


def purity(
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    P(rho) = Tr(rho^2)
    """

    rho = symmetrize_density(rho)

    rho_squared = rho @ rho

    return torch.diagonal(
        rho_squared,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)


def trace_distance(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
) -> torch.Tensor:
    """
    D(rho_a, rho_b)
        = 1/2 ||rho_a - rho_b||_1

    For Hermitian matrices, the trace norm is the sum
    of absolute eigenvalues.
    """

    delta = symmetrize_density(
        rho_a - rho_b
    )

    eigenvalues = torch.linalg.eigvalsh(delta)

    return 0.5 * torch.sum(
        torch.abs(eigenvalues),
        dim=-1,
    )


def fidelity(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Uhlmann fidelity for density matrices.

    F(rho_a, rho_b)
        = [Tr sqrt(
            sqrt(rho_a) rho_b sqrt(rho_a)
          )]^2
    """

    rho_a = symmetrize_density(rho_a)
    rho_b = symmetrize_density(rho_b)

    eig_a, vec_a = torch.linalg.eigh(rho_a)

    eig_a = torch.clamp(
        eig_a,
        min=0.0,
    )

    sqrt_a = (
        vec_a
        @ torch.diag_embed(
            torch.sqrt(
                eig_a.clamp_min(eps)
            )
        )
        @ vec_a.transpose(-1, -2)
    )

    middle = (
        sqrt_a
        @ rho_b
        @ sqrt_a
    )

    middle = symmetrize_density(middle)

    eig_middle = torch.linalg.eigvalsh(
        middle
    )

    eig_middle = torch.clamp(
        eig_middle,
        min=0.0,
    )

    root_fidelity = torch.sum(
        torch.sqrt(
            eig_middle.clamp_min(eps)
        ),
        dim=-1,
    )

    return torch.clamp(
        root_fidelity ** 2,
        min=0.0,
        max=1.0,
    )


def sadi(
    rho_audio: torch.Tensor,
    rho_text: torch.Tensor,
) -> torch.Tensor:
    """
    Semantic-Affective Discordance Index.

        SADI = 1 - Fidelity

    0 -> identical states
    1 -> maximally different / orthogonal states
    """

    return 1.0 - fidelity(
        rho_audio,
        rho_text,
    )

# Descriptor extraction

def compute_v4_quantum_descriptors(
    rho_audio: torch.Tensor,
    rho_text: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute the quantum-inspired descriptors used by V4.
    """

    audio_entropy = von_neumann_entropy(
        rho_audio
    )

    text_entropy = von_neumann_entropy(
        rho_text
    )

    audio_purity = purity(
        rho_audio
    )

    text_purity = purity(
        rho_text
    )

    fidelity_score = fidelity(
        rho_audio,
        rho_text,
    )

    discordance = trace_distance(
        rho_audio,
        rho_text,
    )

    semantic_affective_discordance = (
        1.0 - fidelity_score
    )

    return {
        "audio_entropy": audio_entropy,
        "text_entropy": text_entropy,
        "audio_purity": audio_purity,
        "text_purity": text_purity,
        "fidelity": fidelity_score,
        "trace_distance": discordance,
        "sadi": semantic_affective_discordance,
    }