"""Freeze original spatial block assignments for a BT inner-validation pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def prepare(split_dir, output, outer=1, inner=2):
    if outer == inner:
        raise ValueError("Outer and inner blocks must differ.")
    paths = [split_dir / f"new_no_coral_split_4_fold_{fold:02d}.csv" for fold in range(1, 5)]
    folds = []
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assignment = {r["spygen_code"]: r["set"] for r in rows}
        if len(rows) != len(assignment) or set(assignment.values()) != {"train", "validation"}:
            raise ValueError(f"Invalid fold: {path}")
        folds.append(assignment)
    ids = set(folds[0])
    if any(set(fold) != ids for fold in folds):
        raise ValueError("Fold sample universes differ.")
    records = []
    for code in sorted(ids):
        blocks = [i + 1 for i, fold in enumerate(folds) if fold[code] == "validation"]
        if len(blocks) != 1:
            raise ValueError(f"Expected exactly one outer assignment: {code}")
        block = blocks[0]
        role = "outer_eval" if block == outer else "inner_validation" if block == inner else "inner_train"
        records.append(dict(spygen_code=code, spatial_block=block, role=role))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sample_roles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["spygen_code", "spatial_block", "role"])
        writer.writeheader()
        writer.writerows(records)
    protocol = dict(outer_fold=outer, inner_fold=inner,
                    counts={role: sum(r["role"] == role for r in records)
                            for role in ["inner_train", "inner_validation", "outer_eval"]},
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (output / "split_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--inner-fold", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    print(json.dumps(prepare(args.split_dir, args.output, args.outer_fold, args.inner_fold), indent=2))
