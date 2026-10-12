import torch
import json

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.baselines.b2_attention.model import (
    BaselineAttention,
)

from src.baselines.b2_attention.dataset import (
    get_test_loader,
)

from src.utils.metrics import (
    compute_metrics,
)

from src.config.settings import (

    B2_MODEL_DIR,

)

from src.config.settings import (

    B2_RESULTS_PATH,

)

# ---------------------------------------------------
# configuration
# ---------------------------------------------------

MODEL_PATH = B2_MODEL_DIR / "b2_model.pt"

BATCH_SIZE = 16

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

# ---------------------------------------------------
# dataloader
# ---------------------------------------------------

test_loader = get_test_loader(

    batch_size=BATCH_SIZE,

)

# ---------------------------------------------------
# model
# ---------------------------------------------------

model = BaselineAttention()

model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE,

    )

)

model.to(DEVICE)

model.eval()

# ---------------------------------------------------
# evaluation
# ---------------------------------------------------

predictions = []

targets = []

with torch.no_grad():

    for batch in test_loader:

        embeddings = batch["embedding_sequence"].to(

            DEVICE

        )

        attention_mask = batch["attention_mask"].to(

            DEVICE

        )

        labels = batch["label"].to(

            DEVICE

        )

        logits = model(

            embeddings,

            attention_mask,

        )

        preds = torch.argmax(

            logits,

            dim=1,

        )

        predictions.extend(

            preds.cpu().numpy()

        )

        targets.extend(

            labels.cpu().numpy()

        )

# ---------------------------------------------------
# metrics
# ---------------------------------------------------

metrics = compute_metrics(

    targets,

    predictions,

)

results = {

    "model": "Baseline 2 Attention",

    "dataset": "eRisk",

    "epochs": 30,

    "batch_size": 16,

    "learning_rate": 1e-4,

    "accuracy": metrics["accuracy"],

    "precision": metrics["precision"],

    "recall": metrics["recall"],

    "f1": metrics["f1"],

    "macro_f1": metrics["macro_f1"],

}

with open(

    B2_RESULTS_PATH,

    "w",

) as f:

    json.dump(

        results,

        f,

        indent=4,

    )

print()

print("=" * 60)

print("Baseline 2 Results")

print("=" * 60)

print()

print(

    classification_report(

        targets,

        predictions,

        digits=4,

        zero_division=0,

    )

)

print()

print(f"Accuracy : {metrics['accuracy']:.4f}")

print(f"Precision: {metrics['precision']:.4f}")

print(f"Recall   : {metrics['recall']:.4f}")

print(f"F1       : {metrics['f1']:.4f}")

print(f"Macro-F1 : {metrics['macro_f1']:.4f}")

print()

print(

    confusion_matrix(

        targets,

        predictions,

    )

)