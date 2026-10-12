import torch

def covariance_density_matrix(sequence):

    # sequence:
    # [batch, seq_len, latent_dim]

    centered = (
        sequence -
        sequence.mean(
            dim=1,
            keepdim=True,
        )
    )

    cov = torch.matmul(
        centered.transpose(1, 2),
        centered,
    )

    cov = cov / (
        sequence.shape[1] - 1
    )

    trace = torch.diagonal(
        cov,
        dim1=-2,
        dim2=-1,
    ).sum(-1)

    trace = trace.unsqueeze(-1).unsqueeze(-1)

    rho = cov / (trace + 1e-8)

    return rho