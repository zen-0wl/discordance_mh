import pickle

import torch

import torch.optim as optim

from src.baselines.b1_transformer.model import (
    BaselineTransformer,
)

from src.baselines.b1_transformer.dataset import (
    get_train_loader,
    get_validation_loader,
)

from src.utils.losses import (
    get_loss,
)

from src.utils.trainer import (
    Trainer,
)

from src.config.settings import (

    B1_MODEL_DIR,

)

# ---------------------------------------------------
# configuration
# ---------------------------------------------------

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EPOCHS = 30

PATIENCE = 5

MODEL_PATH = B1_MODEL_DIR / "b1_model.pt"

HISTORY_PATH = B1_MODEL_DIR / "history.pkl"

# ---------------------------------------------------
# device
# ---------------------------------------------------

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print()

print("=" * 60)

print(f"Running on {DEVICE}")

if torch.cuda.is_available():

    print(torch.cuda.get_device_name(0))

print("=" * 60)

# ---------------------------------------------------
# dataloaders
# ---------------------------------------------------

train_loader = get_train_loader(

    batch_size=BATCH_SIZE,

)

val_loader = get_validation_loader(

    batch_size=BATCH_SIZE,

)

# ---------------------------------------------------
# model
# ---------------------------------------------------

model = BaselineTransformer()

# ---------------------------------------------------
# optimizer
# ---------------------------------------------------

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

)

# ---------------------------------------------------
# scheduler
# ---------------------------------------------------

scheduler = optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS,

)

# ---------------------------------------------------
# loss
# ---------------------------------------------------

criterion = get_loss(

    "cross_entropy"

)

# ---------------------------------------------------
# trainer
# ---------------------------------------------------

trainer = Trainer(

    model=model,

    optimizer=optimizer,

    criterion=criterion,

    scheduler=scheduler,

    device=DEVICE,

)

# ---------------------------------------------------
# training
# ---------------------------------------------------

history = trainer.fit(

    train_loader=train_loader,

    val_loader=val_loader,

    epochs=EPOCHS,

    patience=PATIENCE,

)

# ---------------------------------------------------
# save model
# ---------------------------------------------------

torch.save(

    model.state_dict(),

    MODEL_PATH,

)

# ---------------------------------------------------
# save history
# ---------------------------------------------------

with open(

    HISTORY_PATH,

    "wb",

) as f:

    pickle.dump(

        history,

        f,

    )

print()

print("=" * 60)

print("Training Complete")

print("=" * 60)

print()

print(f"Model saved   : {MODEL_PATH}")

print(f"History saved : {HISTORY_PATH}")