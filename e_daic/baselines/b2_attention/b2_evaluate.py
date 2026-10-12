
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from src.baselines.dataset import MultimodalSequenceDataset, collate_fn
from src.baselines.b2_attention.b2_model import BaselineAttention
MODEL_CLASS = BaselineAttention
from src.baselines.b2_attention.b2_model import BaselineAttention

TEST_PATH = Path("data/processed/quantum/test.pkl")
CHECKPOINT = Path("models/b2_best.pt")
OUTPUT_DIR = Path("data/results/b2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 8

def aggregate_by_participant(probabilities, labels, participant_ids):
    probs, labs = {}, {}
    for p,y,pid in zip(probabilities,labels,participant_ids):
        pid=str(pid); probs.setdefault(pid,[]).append(float(p)); labs.setdefault(pid,[]).append(int(y))
    ids=sorted(probs); pp=[]; yy=[]
    for pid in ids:
        ys=set(labs[pid])
        if len(ys)!=1: raise ValueError(f"Inconsistent labels for participant {pid}: {ys}")
        pp.append(float(np.mean(probs[pid]))); yy.append(next(iter(ys)))
    return np.asarray(pp),np.asarray(yy),ids

def metrics(probabilities, labels, threshold):
    pred=(probabilities>=threshold).astype(int)
    out={"macro_f1":float(f1_score(labels,pred,average="macro",zero_division=0)),
         "accuracy":float(accuracy_score(labels,pred)),
         "precision":float(precision_score(labels,pred,zero_division=0)),
         "recall":float(recall_score(labels,pred,zero_division=0)),
         "roc_auc":float(roc_auc_score(labels,probabilities)) if len(np.unique(labels))==2 else float("nan")}
    return out

def bootstrap_ci(probabilities, labels, threshold, metric_name, n=2000, seed=42):
    rng=np.random.default_rng(seed); values=[]; size=len(labels)
    for _ in range(n):
        idx=rng.integers(0,size,size=size); y=labels[idx]; p=probabilities[idx]
        if metric_name=="roc_auc" and len(np.unique(y))<2: continue
        values.append(metrics(p,y,threshold)[metric_name])
    return (float(np.percentile(values,2.5)),float(np.percentile(values,97.5))) if values else (float("nan"),float("nan"))

def parse_args():
    p=argparse.ArgumentParser(description="Evaluate B2_ATTENTION on held-out E-DAIC TEST.")
    p.add_argument("--batch-size",type=int,default=BATCH_SIZE); p.add_argument("--bootstrap",type=int,default=2000)
    return p.parse_args()

def main():
    args=parse_args()
    if not CHECKPOINT.exists(): raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}\nRun training first.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=MultimodalSequenceDataset(TEST_PATH)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn,pin_memory=device.type=="cuda")
    ckpt=torch.load(CHECKPOINT,map_location=device,weights_only=False)
    threshold=float(ckpt["dev_threshold"])
    model=MODEL_CLASS().to(device); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    records=[]
    with torch.no_grad():
        for batch in loader:
            out=model(batch["audio"].to(device),batch["text"].to(device),
                      batch["audio_mask"].to(device),batch["text_mask"].to(device))
            probs=torch.softmax(out["logits"],-1)[:,1].cpu().numpy()
            labels=batch["labels"].numpy()
            for i,p in enumerate(probs):
                records.append({"participant_id":batch["participant_ids"][i],
                                "utterance_id":batch["utterance_ids"][i],
                                "probability":float(p),"label":int(labels[i]),
                                "phq8_total":float(batch["phq8_totals"][i])})
    (OUTPUT_DIR/"segment_predictions.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    p=np.asarray([r["probability"] for r in records]); y=np.asarray([r["label"] for r in records])
    ids=[r["participant_id"] for r in records]
    pp,py,pids=aggregate_by_participant(p,y,ids)
    m=metrics(pp,py,threshold)
    f1ci=bootstrap_ci(pp,py,threshold,"macro_f1",args.bootstrap)
    auc_ci=bootstrap_ci(pp,py,threshold,"roc_auc",args.bootstrap)
    participant_records=[{"participant_id":pid,"probability":float(prob),
                          "prediction":int(prob>=threshold),"label":int(label)}
                         for pid,prob,label in zip(pids,pp,py)]
    (OUTPUT_DIR/"participant_predictions.json").write_text(json.dumps(participant_records,indent=2),encoding="utf-8")
    results={"model":"B2_ATTENTION","checkpoint":str(CHECKPOINT),"checkpoint_epoch":ckpt.get("epoch"),
             "seed":ckpt.get("seed"),"dev_threshold":threshold,"test_records":len(records),
             "test_participants":len(pids),"participant_level_metrics":m,
             "bootstrap_95_ci":{"macro_f1":{"lower":f1ci[0],"upper":f1ci[1]},
                                "roc_auc":{"lower":auc_ci[0],"upper":auc_ci[1]}},
             "protocol":{"model_selection":"DEV","threshold_selection":"DEV",
                         "final_evaluation":"held-out TEST","aggregation":"participant-level mean probability"}}
    path=OUTPUT_DIR/"test_results.json"; path.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("="*60); print("B2_ATTENTION TEST EVALUATION"); print("="*60)
    print(f"Participants: {len(pids)}\nSegments: {len(records)}\nDEV threshold: {threshold:.3f}\n")
    print(f"Macro-F1: {m['macro_f1']:.4f}")
    print(f"Macro-F1 95% CI: [{f1ci[0]:.4f}, {f1ci[1]:.4f}]")
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"ROC-AUC 95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    print(f"Accuracy: {m['accuracy']:.4f}\nPrecision: {m['precision']:.4f}\nRecall: {m['recall']:.4f}")
    print(f"Results: {path}\n"+"="*60)

if __name__=="__main__": main()
