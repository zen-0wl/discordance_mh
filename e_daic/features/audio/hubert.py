import torch
import librosa
import numpy as np

from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
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

MODEL_NAME = "facebook/hubert-base-ls960"

feature_extractor = (
    Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME
    )
)

model = HubertModel.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()


def extract_hubert_embedding(
    audio,
    sample_rate,
):
    """
    Extract HuBERT embedding from waveform.
    """

    # ---------------------------------------------------
    # resample to 16k if needed
    # ---------------------------------------------------

    if sample_rate != 16000:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )

        sample_rate = 16000

    # ---------------------------------------------------
    # tokenize audio
    # ---------------------------------------------------

    inputs = feature_extractor(

        audio,

        sampling_rate=sample_rate,

        return_tensors="pt",

        padding=True,
    )

    # ---------------------------------------------------
    # move to gpu
    # ---------------------------------------------------

    input_values = (
        inputs.input_values.to(device)
    )

    # ---------------------------------------------------
    # inference
    # ---------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_values
        )

    # ---------------------------------------------------
    # hidden states
    # ---------------------------------------------------

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