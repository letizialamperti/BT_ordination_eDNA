"""Frozen softmax32 refit and outer scoring; no inner/outer checkpoint selection."""
import argparse
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from final_checks import sha, read_roles, run_lock, atomic_json


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[1,2,3,4], required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with run_lock(args.output, ".classification.lock"):
        evaluate(args)


def evaluate(args):
    recipe = json.loads((HERE / "frozen_configuration.json").read_text())
    cfg = recipe["classifier"]
    roles_path = HERE / f"splits/fold_{args.fold:02d}.csv"
    roles = read_roles(roles_path, args.fold).set_index("spygen_code")
    frame = pd.read_csv(args.embeddings).sort_values("Sample").reset_index(drop=True)
    if not frame.Sample.is_unique or set(frame.Sample) != set(roles.index):
        raise ValueError("Embedding cohort differs from frozen 411 IDs.")
    if frame.role.tolist() != roles.role.reindex(frame.Sample).tolist():
        raise ValueError("Embedding train/outer assignments differ from frozen spatial split.")
    export_path = args.embeddings.with_name("export_protocol.json")
    if export_path.exists():
        export_protocol = json.loads(export_path.read_text())
        if export_protocol["roles_sha256"] != sha(roles_path) or export_protocol["completed_epochs"] != 6:
            raise ValueError("Export protocol differs from frozen outer roles or duration.")
    protocol = dict(fold=args.fold, classifier=cfg, representation=recipe["representation"],
                    embeddings_sha256=sha(args.embeddings), metadata_sha256=sha(args.metadata),
                    roles_sha256=sha(roles_path), frozen_recipe_sha256=sha(HERE / "frozen_configuration.json"),
                    evaluator_sha256=sha(__file__), outer_scoring=True,
                    checkpoint_selection=False, early_stopping=False,
                    probability_ensemble_saved_but_not_used_for_seed_mean=True)
    record = args.output / "classification_protocol.json"
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError("Existing output protocol differs; preserve it and use a separate output folder.")
    atomic_json(record, protocol)
    features = [f"Dim{i}" for i in range(1,257)]
    values = frame[features].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite preprojector embedding.")
    tr = frame.role.eq("train").to_numpy(); outer = ~tr
    scaler = StandardScaler().fit(values[tr])
    x = scaler.transform(values).astype("float32")
    joblib.dump(scaler, args.output / "input_transform.joblib")
    metadata = pd.read_csv(args.metadata, sep=";").set_index("spygen_code")
    if not metadata.index.is_unique:
        raise ValueError("Duplicate metadata sample IDs.")
    labels = pd.to_numeric(metadata.loc[frame.Sample, "Index_MPA_num"], errors="raise")
    if not labels.isin([1,2,3,4,5]).all():
        raise ValueError("Unexpected/missing protection labels.")
    index = labels.replace({4:3}).to_numpy()
    targets = dict(unprotected_vs_protected=(index>1).astype("int64"),
                   protection=(index==5).astype("int64"),
                   Index_MPA_num=pd.Series(index).map({1:0,2:1,3:2,5:3}).to_numpy(dtype="int64"))
    torch.set_num_threads(1)
    scores, recalls, matrices = [], [], {}
    for target in recipe["targets"]:
        y = targets[target];classes = 4 if target == "Index_MPA_num" else 2
        counts = np.bincount(y[tr],minlength=classes)
        if not (counts>0).all():
            raise ValueError(f"Training lacks a class: {target}")
        weights = 1/counts;weights = weights/weights.mean()
        target_dir = args.output / target;target_dir.mkdir(exist_ok=True)
        probabilities=[]
        for seed in cfg["seeds"]:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(256,cfg["hidden"]),nn.ReLU(),nn.Dropout(cfg["dropout"]),nn.Linear(cfg["hidden"],classes))
            optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
            loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(y[tr])),
                                batch_size=cfg["batch_size"],shuffle=True)
            model.train()
            for _ in range(cfg["max_epochs"]):
                for a,b in loader:
                    optimizer.zero_grad();loss=criterion(model(a),b)
                    if not torch.isfinite(loss):raise FloatingPointError("Non-finite classifier training loss")
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"]);optimizer.step()
            model.eval()
            with torch.inference_mode():
                logits=model(torch.from_numpy(x[outer]));pred=logits.argmax(1).numpy();prob=logits.softmax(1).numpy()
            probabilities.append(prob)
            scores.append(dict(fold=args.fold,target=target,seed=seed,n_train=int(tr.sum()),n_outer=int(outer.sum()),
                               balanced_accuracy=float(balanced_accuracy_score(y[outer],pred)),
                               accuracy=float(accuracy_score(y[outer],pred)),
                               macro_f1=float(f1_score(y[outer],pred,average="macro",labels=list(range(classes)),zero_division=0))))
            prediction=pd.DataFrame(dict(Sample=frame.loc[outer,"Sample"].to_numpy(),label=y[outer],prediction=pred))
            for c in range(classes):prediction[f"probability_{c}"]=prob[:,c]
            prediction.to_csv(target_dir/f"predictions_outer_seed{seed}.csv",index=False)
            precision,recall,f1,support=precision_recall_fscore_support(y[outer],pred,labels=list(range(classes)),zero_division=0)
            for c in range(classes):recalls.append(dict(target=target,seed=seed,class_id=c,n_outer=int(support[c]),precision=float(precision[c]),recall=float(recall[c]),f1=float(f1[c])))
            matrices[f"{target}_seed{seed}"]=confusion_matrix(y[outer],pred,labels=list(range(classes))).tolist()
            torch.save(dict(state_dict=model.state_dict(),target=target,seed=seed,fold=args.fold,
                            classifier=cfg,input_features=features,n_classes=classes,class_weights=weights.tolist()),
                       target_dir/f"final_state_seed{seed}.pt")
        mean_prob=np.mean(probabilities,axis=0)
        ensemble=pd.DataFrame(dict(Sample=frame.loc[outer,"Sample"].to_numpy(),label=y[outer],prediction=mean_prob.argmax(1)))
        for c in range(classes):ensemble[f"probability_{c}"]=mean_prob[:,c]
        ensemble.to_csv(target_dir/"predictions_outer_ensemble3.csv",index=False)
    scores=pd.DataFrame(scores);scores.to_csv(args.output/"outer_scores_by_seed.csv",index=False)
    summary=scores.groupby("target").agg(ba_mean=("balanced_accuracy","mean"),ba_sd=("balanced_accuracy","std"),
        accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),macro_f1_mean=("macro_f1","mean"),macro_f1_sd=("macro_f1","std"))
    summary.to_csv(args.output/"outer_summary.csv")
    pd.DataFrame(recalls).to_csv(args.output/"class_metrics_by_seed.csv",index=False)
    atomic_json(args.output/"confusion_matrices.json",matrices)
    print((100*summary).round(2).to_string())


if __name__=="__main__":main()
