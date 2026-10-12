from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.baselines.b1_transformer.model import TransformerFusion
from src.utils.dataset_loader import get_dataloaders
from src.utils.trainer import Trainer


ROOT = Path(__file__).resolve().parents[3]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=32,
    )

    model = TransformerFusion()
    
    model = model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        criterion=criterion,

        device=DEVICE,

    )

    trainer.fit(

        train_loader=train_loader,

        val_loader=val_loader,

        epochs=50,

        save_dir=ROOT / "models" / "b1_transformer",

        model_name="b1_model.pt",

    )


if __name__ == "__main__":
    main()