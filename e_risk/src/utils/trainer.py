import copy
import numpy as np

import torch

from tqdm import tqdm

from src.utils.metrics import compute_metrics


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.scheduler = scheduler

        self.device = device

        self.model.to(device)

    # ---------------------------------------------------------
    # one training epoch
    # ---------------------------------------------------------

    def train_epoch(self, loader):

        self.model.train()

        running_loss = 0.0

        for batch in tqdm(loader, leave=False):

            embeddings = batch["embedding_sequence"].to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)

            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(

            embeddings,

            attention_mask,

        )

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = self.criterion(

            logits,
            labels,

        )

        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()

        return running_loss / len(loader)

    # ---------------------------------------------------------
    # validation
    # ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader):

        self.model.eval()

        running_loss = 0.0

        predictions = []

        targets = []

        for batch in loader:

            embeddings = batch["embedding_sequence"].to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)

            labels = batch["label"].to(self.device)

            outputs = self.model(
                embeddings,
                attention_mask,
            )

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(

                logits,

                labels,

            )

            running_loss += loss.item()

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

        metrics = compute_metrics(

            targets,

            predictions,

        )

        metrics["loss"] = running_loss / len(loader)

        return metrics

    # ---------------------------------------------------------
    # full training
    # ---------------------------------------------------------

    def fit(

        self,

        train_loader,

        val_loader,

        epochs=30,

        patience=5,

    ):

        history = {

            "train_loss": [],

            "val_loss": [],

            "macro_f1": [],

        }

        best_score = -np.inf

        best_state = None

        wait = 0

        for epoch in range(epochs):

            print()

            print("=" * 60)

            print(f"Epoch {epoch+1}/{epochs}")

            print("=" * 60)

            train_loss = self.train_epoch(

                train_loader

            )

            metrics = self.evaluate(

                val_loader

            )

            history["train_loss"].append(

                train_loss

            )

            history["val_loss"].append(

                metrics["loss"]

            )

            history["macro_f1"].append(

                metrics["macro_f1"]

            )

            print()

            print(

                f"Train Loss : {train_loss:.4f}"

            )

            print(

                f"Val Loss   : {metrics['loss']:.4f}"

            )

            print(

                f"Accuracy   : {metrics['accuracy']:.4f}"

            )

            print(

                f"Precision  : {metrics['precision']:.4f}"

            )

            print(

                f"Recall     : {metrics['recall']:.4f}"

            )

            print(

                f"F1         : {metrics['f1']:.4f}"

            )

            print(

                f"Macro-F1   : {metrics['macro_f1']:.4f}"

            )

            # scheduler

            if self.scheduler is not None:

                self.scheduler.step()

            # early stopping

            if metrics["macro_f1"] > best_score:

                best_score = metrics["macro_f1"]

                wait = 0

                best_state = copy.deepcopy(

                    self.model.state_dict()

                )

            else:

                wait += 1

                print(

                    f"No improvement ({wait}/{patience})"

                )

                if wait >= patience:

                    print()

                    print("Early stopping.")

                    break

        if best_state is not None:

            self.model.load_state_dict(

                best_state

            )

            print()

            print(

                f"Best Validation Macro-F1: {best_score:.4f}"

            )

        return history

    # ---------------------------------------------------------
    # test
    # ---------------------------------------------------------

    @torch.no_grad()
    def test(

        self,

        loader,

    ):

        metrics = self.evaluate(

            loader

        )

        return metrics