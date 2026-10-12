import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
)

# ---------------------------------------------------
# device
# ---------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------------------------
# model
# ---------------------------------------------------

MODEL_NAME = (
    "microsoft/deberta-v3-base"
)

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()


def extract_deberta_embedding(
    text,
):
    """
    Extract DeBERTa embedding.
    """

    # ---------------------------------------------------
    # empty text safety
    # ---------------------------------------------------

    if (
        text is None or
        len(text.strip()) == 0
    ):

        return np.zeros(
            768,
            dtype=np.float32,
        )

    # ---------------------------------------------------
    # tokenize
    # ---------------------------------------------------

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=128,
    )

    # ---------------------------------------------------
    # move tensors to gpu
    # ---------------------------------------------------

    inputs = {

        k: v.to(device)

        for k, v in inputs.items()
    }

    # ---------------------------------------------------
    # inference
    # ---------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    hidden_states = (
        outputs.last_hidden_state
    )

    # ---------------------------------------------------
    # mean pooling
    # ---------------------------------------------------

    embedding = hidden_states.mean(
        dim=1
    ).squeeze()

    embedding = (
        embedding
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return embedding