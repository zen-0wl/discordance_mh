import torch

import torch.nn.functional as F


# ---------------------------------------------------
# density matrix
# ---------------------------------------------------

def build_attention_density(

    attention_vector,

    context_vector,

):

    # --------------------------------------------
    # normalize
    # --------------------------------------------

    attention_vector = F.normalize(

        attention_vector,

        p=2,

        dim=-1,

    )

    context_vector = F.normalize(

        context_vector,

        p=2,

        dim=-1,

    )

    # --------------------------------------------
    # combine
    # --------------------------------------------

    state = 0.5 * (

        attention_vector +

        context_vector

    )

    state = F.normalize(

        state,

        p=2,

        dim=-1,

    )

    # --------------------------------------------
    # density matrix
    # --------------------------------------------

    density = torch.einsum(

        "bi,bj->bij",

        state,

        state,

    )

    return density