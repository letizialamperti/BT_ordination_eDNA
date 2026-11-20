"""Freeze the 411-sample cohort and two inner rotations before training."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pilot_checks import read_roles, atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-roles", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "splits")
    args = parser.parse_args()
    reference = read_roles(args.reference_roles)
    ids = pd.read_csv(args.cohort_csv, usecols=["Sample"], dtype=str).Sample
    if len(ids) != 411 or not ids.is_unique or ids.isna().any():
        raise ValueError("The common cohort must contain exactly 411 unique IDs.")
    if not set(ids).issubset(reference.spygen_code):
        raise ValueError("Cohort IDs absent from original spatial roles.")
    cohort = reference.loc[reference.spygen_code.isin(ids), ["spygen_code", "spatial_block"]].sort_values("spygen_code")
    if len(cohort) != 411 or set(cohort.spatial_block) != {1, 2, 3, 4}:
        raise ValueError("Incomplete spatial cohort.")
    protocol = dict(cohort_size=411, outer_fold=1, inner_folds=[2, 3], encoder_seed=0,
                    candidates=[dict(views="fixed", epochs=2), dict(views="shuffled", epochs=6)],
                    encoder_representation_dim=256, projector_dim=64,
                    classifier=dict(epochs=20, seeds=[0, 1, 2], model="standardized softmax32",
                                    class_weights="inverse training frequency normalized mean1"),
                    targets=["unprotected_vs_protected", "protection", "Index_MPA_num"],
                    selection=dict(primary="mean inner balanced accuracy, equal weight per target and rotation, after averaging classifier seeds",
                                   exact_tie="fixed2, lower computational cost",
                                   report="target-wise and rotation-wise differences and classifier-seed variability",
                                   outer_scoring=False, exploratory=True),
                    cohort_source=str(args.cohort_csv), cohort_source_sha256=sha(args.cohort_csv),
                    spatial_roles_source=str(args.reference_roles), spatial_roles_sha256=sha(args.reference_roles),
                    rotations={})
    args.output.mkdir(parents=True, exist_ok=True)
    # Never change an already generated cohort/assignment silently.
    destinations = {args.output / "cohort_411.csv": cohort}
    for inner in [2, 3]:
        roles = cohort.copy()
        roles["role"] = roles.spatial_block.map(lambda b: "outer_eval" if b == 1 else
                                               "inner_validation" if b == inner else "inner_train")
        classifier = roles.loc[roles.role.ne("outer_eval")].copy()
        directory = args.output / f"outer_01_inner_{inner:02d}"
        destinations[directory / "sample_roles.csv"] = roles
        destinations[directory / "classifier_roles.csv"] = classifier
        protocol["rotations"][str(inner)] = dict(counts=roles.role.value_counts().to_dict(),
                                               training_blocks=[b for b in [2, 3, 4] if b != inner])
    for path, frame in destinations.items():
        if path.exists() and not pd.read_csv(path).equals(frame.reset_index(drop=True)):
            raise ValueError(f"Existing assignments differ: {path}")
    record = args.output.parent / "experiment_protocol.json"
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError("Existing experiment protocol differs; use a separate bundle.")
    for path, frame in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        if path.name in {"sample_roles.csv", "classifier_roles.csv"}:
            read_roles(path, include_outer=path.name == "sample_roles.csv")
    atomic_json(record, protocol)
    print(json.dumps(protocol["rotations"], indent=2))


if __name__ == "__main__":
    main()
