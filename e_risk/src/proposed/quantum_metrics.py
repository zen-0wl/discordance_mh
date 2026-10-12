import torch


# ---------------------------------------------------
# Von Neumann Entropy
# ---------------------------------------------------

def von_neumann_entropy(

    density,

    eps=1e-8,

):

    eigenvalues = torch.linalg.eigvalsh(

        density

    )

    eigenvalues = torch.clamp(

        eigenvalues,

        min=eps,

    )

    entropy = -(

        eigenvalues *

        torch.log2(

            eigenvalues

        )

    ).sum(

        dim=1

    )

    return entropy


# ---------------------------------------------------
# Trace Distance
# ---------------------------------------------------

def trace_distance(

    density_a,

    density_b,

):

    difference = (

        density_a -

        density_b

    )

    singular_values = torch.linalg.svdvals(

        difference

    )

    return 0.5 * singular_values.sum(

        dim=1

    )


# ---------------------------------------------------
# Fidelity
# ---------------------------------------------------

def fidelity(

    density_a,

    density_b,

):

    product = torch.bmm(

        density_a,

        density_b,

    )

    trace = torch.diagonal(

        product,

        dim1=1,

        dim2=2,

    ).sum(

        dim=1

    )

    return trace


# ---------------------------------------------------
# Purity
# ---------------------------------------------------

def purity(

    density,

):

    product = torch.bmm(

        density,

        density,

    )

    return torch.diagonal(

        product,

        dim1=1,

        dim2=2,

    ).sum(

        dim=1

    )