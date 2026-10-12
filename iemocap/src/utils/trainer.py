from pathlib import Path

import json
import pickle

import torch

from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer=None,
        criterion=None,
        device="cuda",
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device
        
        self.history = {

            "train_loss": [],
            "val_loss": [],

            "train_accuracy": [],
            "val_accuracy": [],

            "train_precision": [],
            "val_precision": [],

            "train_recall": [],
            "val_recall": [],

            "train_macro_f1": [],
            "val_macro_f1": [],

        }

    # -------------------------------------------------

    def train_epoch(
        self,
        dataloader,
    ):

        self.model.train()

        running_loss = 0

        predictions = []

        labels = []

        progress = tqdm(
            dataloader,
            desc="Training",
            leave=False,
        )

        for batch in progress:

            audio = batch["audio"].to(self.device)

            text = batch["text"].to(self.device)

            target = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(
                audio,
                text,
            )

            loss = self.criterion(
                logits,
                target,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            pred = logits.argmax(dim=1)

            predictions.extend(
                pred.cpu().numpy()
            )

            labels.extend(
                target.cpu().numpy()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        metrics = self.compute_metrics(
            labels,
            predictions,
        )

        metrics["loss"] = (
            running_loss /
            len(dataloader)
        )

        return metrics

    # -------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
    ):

        self.model.eval()

        running_loss = 0

        predictions = []

        labels = []

        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            leave=False,
        ):

            audio = batch["audio"].to(self.device)

            text = batch["text"].to(self.device)

            target = batch["label"].to(self.device)

            logits = self.model(
                audio,
                text,
            )

            loss = self.criterion(
                logits,
                target,
            )

            running_loss += loss.item()

            pred = logits.argmax(dim=1)

            predictions.extend(
                pred.cpu().numpy()
            )

            labels.extend(
                target.cpu().numpy()
            )

        metrics = self.compute_metrics(
            labels,
            predictions,
        )

        metrics["loss"] = (
            running_loss /
            len(dataloader)
        )

        return metrics

    def save_model(
        self,
        path,
    ):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.model.state_dict(),
            path,
        )

    # -------------------------------------------------

    def save_history(
        self,
        path,
    ):

        with open(
            path,
            "wb",
        ) as f:

            pickle.dump(
                self.history,
                f,
            )

    # -------------------------------------------------

    def save_results(
        self,
        results,
        path,
    ):

        with open(
            path,
            "w",
        ) as f:

            json.dump(
                results,
                f,
                indent=4,
            )

    # -------------------------------------------------
    # training loop
    # -------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        save_dir,
        model_name,
        history_name="history.pkl",
        results_name="results.json",
    ):

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            save_dir /
            model_name
        )

        history_path = (
            save_dir /
            history_name
        )

        results_path = (
            save_dir /
            results_name
        )

        best_macro_f1 = -1

        best_epoch = 0

        best_results = None

        print()

        print("=" * 60)
        print("Training")
        print("=" * 60)

        for epoch in range(epochs):

            print()

            print(
                f"Epoch {epoch + 1}/{epochs}"
            )

            train_metrics = self.train_epoch(
                train_loader,
            )

            val_metrics = self.evaluate(
                val_loader,
            )

            # -----------------------------
            # history
            # -----------------------------

            self.history["train_loss"].append(
                train_metrics["loss"]
            )

            self.history["val_loss"].append(
                val_metrics["loss"]
            )

            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )

            self.history["val_accuracy"].append(
                val_metrics["accuracy"]
            )

            self.history["train_precision"].append(
                train_metrics["precision"]
            )

            self.history["val_precision"].append(
                val_metrics["precision"]
            )

            self.history["train_recall"].append(
                train_metrics["recall"]
            )

            self.history["val_recall"].append(
                val_metrics["recall"]
            )

            self.history["train_macro_f1"].append(
                train_metrics["macro_f1"]
            )

            self.history["val_macro_f1"].append(
                val_metrics["macro_f1"]
            )

            # -----------------------------
            # console output
            # -----------------------------

            print(
                f"Train Loss : {train_metrics['loss']:.4f}"
            )

            print(
                f"Val Loss   : {val_metrics['loss']:.4f}"
            )

            print(
                f"Val Acc    : {val_metrics['accuracy']:.4f}"
            )

            print(
                f"Val MacroF1: {val_metrics['macro_f1']:.4f}"
            )

            # -----------------------------
            # save best
            # -----------------------------

            if (
                val_metrics["macro_f1"]
                > best_macro_f1
            ):

                best_macro_f1 = (
                    val_metrics["macro_f1"]
                )

                best_epoch = epoch + 1

                best_results = val_metrics.copy()

                self.save_model(
                    model_path,
                )

                print(
                    "✓ Best model updated."
                )

        # ---------------------------------
        # save history/results
        # ---------------------------------

        self.save_history(
            history_path,
        )

        best_results["best_epoch"] = (
            best_epoch
        )

        self.save_results(

            best_results,

            results_path,

        )

        print()

        print("=" * 60)
        print("Training Complete")
        print("=" * 60)

        print()

        print(
            f"Best Epoch     : {best_epoch}"
        )

        print(
            f"Best Macro-F1  : "
            f"{best_results['macro_f1']:.4f}"
        )

        print()

        print(
            f"Model   : {model_path}"
        )

        print(
            f"History : {history_path}"
        )

        print(
            f"Results : {results_path}"
        )

    # -------------------------------------------------

    @staticmethod
    def compute_metrics(
        labels,
        predictions,
    ):

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        return {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "macro_f1": f1,

        }