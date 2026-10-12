import torch
import torch.nn.functional as F


def cosine_discordance(
    audio_state,
    text_state,
):

    cosine = F.cosine_similarity(
        audio_state,
        text_state,
        dim=-1,
    )

    return 1 - cosine


def euclidean_discordance(
    audio_state,
    text_state,
):

    return torch.norm(
        audio_state - text_state,
        dim=-1,
    )


def entropy_discordance(
    audio_state,
    text_state,
):

    eps = 1e-8

    a_prob = F.softmax(
        audio_state,
        dim=-1,
    )

    t_prob = F.softmax(
        text_state,
        dim=-1,
    )

    a_entropy = -torch.sum(
        a_prob * torch.log(a_prob + eps),
        dim=-1,
    )

    t_entropy = -torch.sum(
        t_prob * torch.log(t_prob + eps),
        dim=-1,
    )

    return torch.abs(
        a_entropy - t_entropy
    )