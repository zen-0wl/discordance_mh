import torch


def trace_distance(
    rho_a,
    rho_b,
):

    diff = rho_a - rho_b

    trace_norm = torch.linalg.norm(
        diff,
        ord="nuc",
        dim=(-2, -1),
    )

    return 0.5 * trace_norm


def von_neumann_entropy(
    rho,
    eps=1e-6,
):

    # ---------------------------------------------------
    # numerical stabilization
    # ---------------------------------------------------

    identity = torch.eye(
        rho.shape[-1],
        device=rho.device,
    )

    identity = identity.unsqueeze(0)

    rho = rho + eps * identity

    # ---------------------------------------------------
    # eigenvalues
    # ---------------------------------------------------

    eigenvalues = torch.linalg.eigvalsh(
        rho
    )

    eigenvalues = torch.clamp(
        eigenvalues,
        min=eps,
    )

    entropy = -torch.sum(

        eigenvalues *
        torch.log(eigenvalues),

        dim=-1,
    )

    return entropy


def fidelity(
    rho_a,
    rho_b,
):

    overlap = torch.matmul(
        rho_a,
        rho_b,
    )

    fidelity_score = torch.diagonal(
        overlap,
        dim1=-2,
        dim2=-1,
    ).sum(-1)

    return fidelity_score