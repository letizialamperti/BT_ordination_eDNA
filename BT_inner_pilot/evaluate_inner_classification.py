"""Compare pilot checkpoints by a fixed-budget classifier on excluded inner samples."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    args = parser.parse_args()
    if args.classifier_epochs < 1:
        parser.error("--classifier-epochs must be positive.")
    paths = [args.embeddings_root / "initial/embeddings_preprojector_256D.csv",
             *sorted(args.embeddings_root.glob("epoch_*/embeddings_preprojector_256D.csv"))]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("An initial embedding export and completed epoch exports are required.")
    if len(paths) < 2:
        raise ValueError("Export the initial encoder and at least one trained checkpoint.")
    metadata = pd.read_csv(args.metadata, sep=";").set_index("spygen_code")
    assert metadata.index.is_unique
    features = [f"Dim{i}" for i in range(1, 257)]
    protocol = dict(
        epochs=args.classifier_epochs, seeds=[0,1,2], architecture="256-32-ReLU-dropout0.3-logits",
        lr=.0003, weight_decay=.001, batch_size=16, class_weights="inverse training frequency normalized mean1",
        metadata_sha256=sha(args.metadata),
        embeddings_sha256={str(p): sha(p) for p in paths},
        outer_scoring=False, inner_score_used_to_compare_SS_checkpoints=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / "classifier_protocol.json"
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError("Existing output uses a different protocol or embeddings. Select a fresh output.")
    record.write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(1)
    rows = []
    reference = None
    for path in paths:
        frame = pd.read_csv(path).sort_values("Sample").reset_index(drop=True)
        assert frame.Sample.is_unique and set(frame.role) == {"inner_train", "inner_validation"}
        if reference is None:
            reference = frame[["Sample", "role"]]
        else:
            assert frame[["Sample", "role"]].equals(reference)
        tr = frame.role.eq("inner_train").to_numpy()
        va = ~tr
        values = frame[features].to_numpy(dtype="float32")
        assert np.isfinite(values).all()
        scaler = StandardScaler().fit(values[tr])
        x = scaler.transform(values).astype("float32")
        index = metadata.loc[frame.Sample, "Index_MPA_num"].replace({4: 3}).to_numpy()
        targets = {
            "unprotected_vs_protected": (index > 1).astype("int64"),
            "protection": (index == 5).astype("int64"),
            "Index_MPA_num": pd.Series(index).map({1:0,2:1,3:2,5:3}).to_numpy(dtype="int64"),
        }
        for target, y in targets.items():
            classes = 4 if target == "Index_MPA_num" else 2
            assert set(y[tr]) == set(y[va]) == set(range(classes))
            for seed in [0, 1, 2]:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = nn.Sequential(nn.Linear(256,32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32,classes))
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
                counts = np.bincount(y[tr], minlength=classes)
                weights = 1 / counts
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(), dtype=torch.float32))
                loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr])),
                                    batch_size=16, shuffle=True)
                model.train()
                for _ in range(args.classifier_epochs):
                    for a, b in loader:
                        optimizer.zero_grad()
                        loss = criterion(model(a), b)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                model.eval()
                with torch.inference_mode():
                    prediction = model(torch.from_numpy(x[va])).argmax(1).numpy()
                rows.append(dict(checkpoint=path.parent.name, target=target, seed=seed,
                                 n_train=int(tr.sum()), n_validation=int(va.sum()),
                                 balanced_accuracy=float(balanced_accuracy_score(y[va], prediction)),
                                 accuracy=float(accuracy_score(y[va], prediction)),
                                 macro_f1=float(f1_score(y[va], prediction, average="macro", zero_division=0))))
        pd.DataFrame(rows).to_csv(args.output / "inner_scores_by_seed.csv", index=False)
    scores = pd.DataFrame(rows)
    summary = scores.groupby(["checkpoint", "target"])[["balanced_accuracy", "accuracy", "macro_f1"]].mean()
    summary.to_csv(args.output / "inner_means.csv")
    print((100*summary).round(2).to_string())


if __name__ == "__main__":
    main()
