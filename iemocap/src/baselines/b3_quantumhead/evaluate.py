from pathlib import Path

import json

import torch
import torch.nn as nn

from src.baselines.b3_quantumhead.model import QuantumGatedFusion
from src.utils.dataset_loader import get_dataloaders
from src.utils.trainer import Trainer


ROOT = Path(__file__).resolve().parents[3]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    _, _, test_loader = get_dataloaders(
        batch_size=32,
    )

    model = QuantumGatedFusion()
    
    model = model.to(DEVICE)

    model.load_state_dict(

        torch.load(

            ROOT
            / "models"
            / "b3_quantumhead"
            / "b3_model.pt",

            map_location=DEVICE,

        )

    )
    
    model.eval()

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(

        model=model,

        optimizer=None,

        criterion=criterion,

        device=DEVICE,

    )

    metrics = trainer.evaluate(
        test_loader,
    )

    print()

    print("=" * 60)
    print("Test Results")
    print("=" * 60)

    for key, value in metrics.items():

        print(
            f"{key:12s}: {value:.4f}"
        )

    with open(

        ROOT
        / "models"
        / "b3_quantumhead"
        / "test_results.json",

        "w",

    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()