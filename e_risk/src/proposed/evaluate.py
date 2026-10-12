import json

import torch

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

)

from src.proposed.model import ProposedV4

from src.baselines.b1_transformer.dataset import (

    get_test_loader,

)

from src.config.settings import MODEL_DIR

# ---------------------------------------------------
# paths
# ---------------------------------------------------

SAVE_DIR = MODEL_DIR / "proposed"

MODEL_PATH = SAVE_DIR / "proposed_v4.pt"

RESULT_PATH = SAVE_DIR / "results.json"

# ---------------------------------------------------
# device
# ---------------------------------------------------

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

# ---------------------------------------------------
# model
# ---------------------------------------------------

model = ProposedV4()

model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE,

    )

)

model.to(DEVICE)

model.eval()

# ---------------------------------------------------
# data
# ---------------------------------------------------

loader = get_test_loader(

    batch_size=16,

)

# ---------------------------------------------------
# evaluation
# ---------------------------------------------------

predictions = []

labels = []

with torch.no_grad():

    for batch in loader:

        embeddings = batch["embedding_sequence"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        target = batch["label"].to(DEVICE)

        logits, diagnostics = model(

            embeddings,

            attention_mask,

        )

        probs = torch.softmax(logits, dim=1)

        prediction = (probs[:, 1] >= 0.50).long()

        predictions.extend(

            prediction.cpu().numpy()

        )

        labels.extend(

            target.cpu().numpy()

        )

# ---------------------------------------------------
# metrics
# ---------------------------------------------------

accuracy = accuracy_score(

    labels,

    predictions,

)

precision = precision_score(

    labels,

    predictions,

)

recall = recall_score(

    labels,

    predictions,

)

f1 = f1_score(

    labels,

    predictions,

)

macro_f1 = f1_score(

    labels,

    predictions,

    average="macro",

)

results = {

    "model": "Proposed V4 Discordance",

    "dataset": "eRisk",

    "epochs": 30,

    "batch_size": 16,

    "learning_rate": 1e-4,

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "macro_f1": macro_f1,

}

# ---------------------------------------------------
# save
# ---------------------------------------------------

with open(

    RESULT_PATH,

    "w",

) as f:

    json.dump(

        results,

        f,

        indent=4,

    )

print()

print(json.dumps(

    results,

    indent=4,

))