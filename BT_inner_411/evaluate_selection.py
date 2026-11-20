"""Evaluate the two frozen candidates in both agreed inner rotations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from pilot_checks import atomic_json, run_lock


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Downloaded run tree, containing outer_01_inner_02 and outer_01_inner_03.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=HERE.parents[1] / "Metadata_AMP_460.csv")
    args = parser.parse_args()
    with run_lock(args.output, ".selection.lock"):
        evaluate(args)


def evaluate(args):
    protocol_path = HERE / "experiment_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    # Validate every expected export before starting any classifier fit.
    jobs = []
    for inner in protocol["inner_folds"]:
        for candidate in protocol["candidates"]:
            mode, epochs = candidate["views"], candidate["epochs"]
            run = args.runs_root / f"outer_01_inner_{inner:02d}" / mode / "seed_0"
            root = run / "embeddings"
            config = json.loads((run / "config.json").read_text())
            roles_path = HERE / f"splits/outer_01_inner_{inner:02d}/sample_roles.csv"
            classifier_roles_path = roles_path.with_name("classifier_roles.csv")
            roles = pd.read_csv(roles_path)
            expected_ids = set(roles.loc[roles.role.ne("outer_eval"), "spygen_code"])
            if (config["roles_sha256"] != hashlib.sha256(roles_path.read_bytes()).hexdigest()
                or config["views"] != mode or config["seed"] != 0 or config["max_epochs"] != epochs
                or {Path(p).stem for p in config["raw_input_stats"]} != expected_ids):
                raise ValueError(f"Run cohort or configuration differs: {run}")
            expected = {"initial", f"epoch_{epochs:02d}"}
            actual = {p.parent.name for p in root.glob("*/embeddings_preprojector_256D.csv")}
            if actual != expected:
                raise ValueError(f"Expected only {sorted(expected)}, found {sorted(actual)} at {root}")
            for checkpoint in expected:
                manifest = json.loads((root / checkpoint / "manifest.json").read_text())
                if (manifest["roles_sha256"] != hashlib.sha256(classifier_roles_path.read_bytes()).hexdigest()
                    or manifest["config_sha256"] != hashlib.sha256((run / "config.json").read_bytes()).hexdigest()):
                    raise ValueError(f"Export provenance differs: {root / checkpoint}")
                for name, digest in manifest["output_sha256"].items():
                    if hashlib.sha256((root / checkpoint / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Changed export file: {root / checkpoint / name}")
            jobs.append((inner, mode, epochs, root))
    frames = []
    for inner, mode, epochs, root in jobs:
        output = args.output / f"inner_{inner:02d}_{mode}{epochs}"
        subprocess.run([sys.executable, str(HERE / "evaluate_inner_classification.py"),
                        "--embeddings-root", str(root),
                        "--metadata", str(args.metadata),
                        "--roles", str(HERE / f"splits/outer_01_inner_{inner:02d}/classifier_roles.csv"),
                        "--classifier-epochs", "20", "--output", str(output)], check=True)
        frame = pd.read_csv(output / "inner_scores_by_seed.csv")
        frame["inner_fold"], frame["candidate"] = inner, f"{mode}{epochs}"
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(["inner_fold", "candidate", "checkpoint", "target", "seed"]).any():
        raise ValueError("Duplicate classifier scores.")
    scores.to_csv(args.output / "all_scores_by_seed.csv", index=False)
    trained = scores.loc[scores.checkpoint.ne("initial")]
    expected_n = 2 * 2 * 3 * 3
    if len(trained) != expected_n:
        raise ValueError("Incomplete candidate results.")
    cells = trained.groupby(["candidate", "inner_fold", "target"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        classifier_seed_sd=("balanced_accuracy", "std"),
        accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"))
    cells.to_csv(args.output / "comparison_by_rotation_target.csv", index=False)
    overall = cells.groupby("candidate", as_index=False).balanced_accuracy.mean()
    overall["pretraining_epochs"] = overall.candidate.map({"fixed2": 2, "shuffled6": 6})
    overall = overall.sort_values(["balanced_accuracy", "pretraining_epochs"], ascending=[False, True])
    overall.to_csv(args.output / "candidate_selection.csv", index=False)
    atomic_json(args.output / "selection_result.json", dict(
        recommended_candidate=overall.iloc[0].candidate,
        criterion=protocol["selection"], protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        status="candidate recommendation; inspect target/rotation results before final training",
        outer_scoring=False))
    print("Candidate recommendation:\n", overall.to_string(index=False))


if __name__ == "__main__":
    main()
