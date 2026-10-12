"""
Quantum-inspired metrics for semantic-affective discordance.

These metrics operate on density matrices produced from
the audio and text quantum states.
"""

import torch


EPS = 1e-8


def density_matrix(state: torch.Tensor) -> torch.Tensor:
    """
    Construct density matrix:

        ρ = |ψ><ψ|

    Input:
        state : (B, D)

    Output:
        rho : (B, D, D)
    """

    state = torch.nn.functional.normalize(
        state,
        p=2,
        dim=-1,
    )

    return torch.einsum(
        "bi,bj->bij",
        state,
        state,
    )


def purity(rho: torch.Tensor) -> torch.Tensor:
    """
    Purity:

        Tr(ρ²)

    Returns:
        (B,)
    """

    rho2 = rho @ rho

    return torch.diagonal(
        rho2,
        dim1=-2,
        dim2=-1,
    ).sum(-1)


def trace_distance(
    rho_a,
    rho_b,
):
    """
    Approximate trace distance.

    Uses Frobenius norm for efficiency.

    Returns:
        (B,)
    """

    diff = rho_a - rho_b

    return torch.norm(
        diff,
        dim=(1, 2),
        p="fro",
    )


def fidelity(
    rho_a,
    rho_b,
):
    """
    Lightweight fidelity approximation.

    Returns:
        (B,)
    """

    return torch.real(
    (rho_a * rho_b).sum(dim=(1,2))
)


def von_neumann_entropy(
    rho,
):
    """
    Von Neumann entropy:

        -Tr(ρ log ρ)

    Returns:
        (B,)
    """

    eigvals = torch.linalg.eigvalsh(
        rho
    )

    eigvals = torch.clamp(
        eigvals,
        min=EPS,
    )

    entropy = -(
        eigvals
        * torch.log(
            eigvals
        )
    ).sum(
        dim=-1
    )

    return entropy.real


def entropy_gap(
    rho_a,
    rho_b,
):
    """
    Difference between
    audio/text uncertainty.

    Returns:
        (B,)
    """

    return torch.abs(

        von_neumann_entropy(
            rho_a
        )

        -

        von_neumann_entropy(
            rho_b
        )

    )


def discordance_vector(
    rho_audio,
    rho_text,
):
    """
    Build the complete
    discordance representation.

    Output:
        (B,4)

        [
            fidelity,
            trace_distance,
            purity_gap,
            entropy_gap,
        ]
    """

    fidelity_score = fidelity(
        rho_audio,
        rho_text,
    )

    trace_score = trace_distance(
        rho_audio,
        rho_text,
    )

    purity_gap = torch.abs(

        purity(
            rho_audio
        )

        -

        purity(
            rho_text
        )

    )

    entropy_diff = entropy_gap(
        rho_audio,
        rho_text,
    )

    return torch.stack(
    [
        fidelity_score.real,
        trace_score.real,
        purity_gap.real,
        entropy_diff.real,
    ],
    dim=1,
).float()