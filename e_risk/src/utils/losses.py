import torch

import torch.nn as nn

import torch.nn.functional as F


# ---------------------------------------------------
# Cross Entropy Loss
# ---------------------------------------------------

def get_cross_entropy_loss(

    class_weights=None,

):

    if class_weights is not None:

        class_weights = torch.tensor(

            class_weights,

            dtype=torch.float32,

        )

    return nn.CrossEntropyLoss(

        weight=class_weights,

    )


# ---------------------------------------------------
# Focal Loss
# ---------------------------------------------------

class FocalLoss(nn.Module):

    def __init__(

        self,

        alpha=1.0,

        gamma=2.0,

        reduction="mean",

    ):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.reduction = reduction

    def forward(

        self,

        logits,

        targets,

    ):

        ce = F.cross_entropy(

            logits,

            targets,

            reduction="none",

        )

        pt = torch.exp(-ce)

        loss = self.alpha * (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":

            return loss.mean()

        if self.reduction == "sum":

            return loss.sum()

        return loss


# ---------------------------------------------------
# Loss Factory
# ---------------------------------------------------

def get_loss(

    name="cross_entropy",

    class_weights=None,

):

    if name == "cross_entropy":

        return get_cross_entropy_loss(

            class_weights,

        )

    if name == "focal":

        return FocalLoss()

    raise ValueError(

        f"Unknown loss: {name}"

    )