import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
)

from src.config.settings import (
    PROCESSED_DATA_DIR,
    ERISK_FEATURE_ROOT,
)

# ---------------------------------------------------
# configuration
# ---------------------------------------------------

MODEL_NAME = "microsoft/deberta-v3-base"

MAX_POSTS = 128

MAX_LENGTH = 512

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("=" * 60)
    print("Using GPU")
    print(torch.cuda.get_device_name(0))
    print("=" * 60)

else:

    DEVICE = torch.device("cpu")

    print("=" * 60)
    print("Using CPU")
    print("=" * 60)

# ---------------------------------------------------
# paths
# ---------------------------------------------------

INPUT_FILE = PROCESSED_DATA_DIR / "dataset.pkl"

OUTPUT_FILE = ERISK_FEATURE_ROOT / "embeddings.pkl"

ERISK_FEATURE_ROOT.mkdir(

    parents=True,

    exist_ok=True,
)

# ---------------------------------------------------
# tokenizer + model
# ---------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME

)

print("Loading model...")

model = AutoModel.from_pretrained(

    MODEL_NAME

)

model.to(DEVICE)

model.eval()

print(f"Running on {DEVICE}")

# ---------------------------------------------------
# mean pooling
# ---------------------------------------------------

def mean_pooling(

    output,

    attention_mask,

):

    embeddings = output.last_hidden_state

    mask = (

        attention_mask

        .unsqueeze(-1)

        .expand(embeddings.size())

        .float()
    )

    summed = torch.sum(

        embeddings * mask,

        dim=1,

    )

    counts = torch.clamp(

        mask.sum(dim=1),

        min=1e-9,

    )

    return summed / counts

# ---------------------------------------------------
# load dataset
# ---------------------------------------------------

dataset = pd.read_pickle(

    INPUT_FILE

)

print(f"Subjects: {len(dataset)}")

# ---------------------------------------------------
# extraction
# ---------------------------------------------------

feature_dataset = []

for _, row in tqdm(

    dataset.iterrows(),

    total=len(dataset),

):

    posts = row["posts"][-MAX_POSTS:]

    sequence = []

    for post in posts:

        encoded = tokenizer(

            post,

            max_length=MAX_LENGTH,

            truncation=True,

            padding="max_length",

            return_tensors="pt",

        )

        encoded = {

            k: v.to(DEVICE)

            for k, v in encoded.items()

        }

        with torch.no_grad():

            output = model(**encoded)

            embedding = mean_pooling(

                output,

                encoded["attention_mask"],

            )

        sequence.append(

            embedding

            .squeeze()

            .cpu()

            .numpy()

            .astype(np.float32)
        )
    
    if len(sequence) == 0:
        continue

    sequence = np.stack(sequence)

    feature_dataset.append({

        "subject_id": row["subject_id"],

        "label": row["label"],

        "embedding_sequence": sequence,

        "sequence_length": len(sequence),
        
        "original_posts": row["num_posts"],
        
        "encoder": MODEL_NAME,

    })

# ---------------------------------------------------
# save
# ---------------------------------------------------

with open(

    OUTPUT_FILE,

    "wb",

) as f:

    pickle.dump(

        feature_dataset,

        f,

    )

print()

print("=" * 60)

print("Feature extraction complete.")

print(f"Subjects : {len(feature_dataset)}")

print(f"Saved to : {OUTPUT_FILE}")

print("=" * 60)