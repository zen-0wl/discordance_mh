
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.baselines.dataset import MultimodalSequenceDataset, collate_fn
from src.baselines.b2_attention.b2_model import BaselineAttention
MODEL_CLASS = BaselineAttention
from src.baselines.b2_attention.b2_model import BaselineAttention

TRAIN_PATH = Path("data/processed/quantum/train.pkl")
DEV_PATH = Path("data/processed/quantum/dev.pkl")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("data/results/b2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aggregate_by_participant(probabilities, labels, participant_ids):
    probs, labs = {}, {}
    for p, y, pid in zip(probabilities, labels, participant_ids):
        pid = str(pid)
        probs.setdefault(pid, []).append(float(p))
        labs.setdefault(pid, []).append(int(y))
    ids = sorted(probs)
    out_p, out_y = [], []
    for pid in ids:
        ys = set(labs[pid])
        if len(ys) != 1:
            raise ValueError(f"Inconsistent labels for participant {pid}: {ys}")
        out_p.append(float(np.mean(probs[pid])))
        out_y.append(next(iter(ys)))
    return np.asarray(out_p), np.asarray(out_y), ids

def find_best_threshold(probabilities, labels):
    best_t, best_f1 = 0.50, -float("inf")
    for t in np.linspace(0.05, 0.95, 181):
        pred = (probabilities >= t).astype(int)
        score = f1_score(labels, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1

def calculate_metrics(probabilities, labels, threshold):
    pred = (probabilities >= threshold).astype(int)
    out = {
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
    }
    out["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return out

def evaluate_dev(model, loader, device):
    model.eval()
    ps, ys, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["audio"].to(device), batch["text"].to(device),
                        batch["audio_mask"].to(device), batch["text_mask"].to(device))
            ps.extend(torch.softmax(out["logits"], -1)[:, 1].cpu().numpy().tolist())
            ys.extend(batch["labels"].numpy().tolist())
            pids.extend(batch["participant_ids"])
    return np.asarray(ps), np.asarray(ys), pids

def parse_args():
    p = argparse.ArgumentParser(description="Train B2_ATTENTION.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()

def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0: raise ValueError("epochs and batch-size must be > 0")
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}\nEpochs: {args.epochs}\nBatch size: {args.batch_size}\n")

    train_ds = MultimodalSequenceDataset(TRAIN_PATH)
    dev_ds = MultimodalSequenceDataset(DEV_PATH)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, pin_memory=device.type=="cuda")
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, pin_memory=device.type=="cuda")

    model = MODEL_CLASS().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_f1, best_epoch, best_threshold = -float("inf"), None, 0.50
    best_path = MODEL_DIR / "b2_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0
        for batch in train_loader:
            audio, text = batch["audio"].to(device), batch["text"].to(device)
            am, tm = batch["audio_mask"].to(device), batch["text_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(audio, text, am, tm)["logits"], labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()

        ps, ys, pids = evaluate_dev(model, dev_loader, device)
        pp, py, pids = aggregate_by_participant(ps, ys, pids)
        threshold, _ = find_best_threshold(pp, py)
        metrics = calculate_metrics(pp, py, threshold)
        rec = {"epoch": epoch, "train_loss": total_loss/len(train_loader),
               "dev_threshold": threshold, "dev_participants": len(py), **{f"dev_{k}":v for k,v in metrics.items()}}
        history.append(rec)
        print(f"Epoch {epoch:02d} | Train loss: {rec['train_loss']:.4f} | Dev Macro-F1: {metrics['macro_f1']:.4f} | Dev ROC-AUC: {metrics['roc_auc']:.4f} | Threshold: {threshold:.3f}")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, best_threshold = metrics["macro_f1"], epoch, threshold
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "seed": args.seed, "batch_size": args.batch_size,
                        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                        "dev_threshold": threshold, "dev_metrics": metrics, "model_name": "B2_ATTENTION"}, best_path)
            print(f"  -> New best checkpoint saved: {best_path}")

    history_path = RESULTS_DIR / f"training_history_seed_{args.seed}.json"
    summary_path = RESULTS_DIR / f"training_summary_seed_{args.seed}.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"model":"B2_ATTENTION","seed":args.seed,"epochs_requested":args.epochs,
               "best_epoch":best_epoch,"best_dev_macro_f1":best_f1,"best_dev_threshold":best_threshold,
               "checkpoint":str(best_path),"train_records":len(train_ds),"dev_records":len(dev_ds)}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTraining complete.\nBest epoch: {best_epoch}\nBest Dev Macro-F1: {best_f1:.4f}\nDEV-selected threshold: {best_threshold:.3f}\nCheckpoint: {best_path}")

if __name__ == "__main__":
    main()
