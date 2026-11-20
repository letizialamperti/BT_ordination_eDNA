"""Export fixed 500-pair mean representations for training and inner validation only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import pandas as pd
from pilot_checks import read_roles, run_lock, atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--roles", type=Path, default=here / "splits/classifier_roles.csv")
    parser.add_argument("--checkpoint", default="all", help="all, initial, epoch_01, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    with run_lock(args.run_dir):
        export(args, here)


def export(args, here):
    roles = read_roles(args.roles, include_outer=False)
    config = json.loads((args.run_dir / "config.json").read_text())
    training_roles = args.run_dir / "sample_roles.csv"
    if sha(training_roles) != config["roles_sha256"]:
        raise ValueError("Training roles differ from the frozen configuration.")
    trained = read_roles(training_roles).set_index("spygen_code")
    for row in roles.itertuples():
        if row.spygen_code not in trained.index or any(
            trained.loc[row.spygen_code, col] != getattr(row, col) for col in ["role", "spatial_block"]
        ):
            raise ValueError(f"Export sample or role differs from training: {row.spygen_code}")
    samples = [args.samples_dir / f"{code}.csv" for code in roles.spygen_code]
    source_stats = {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in samples}
    if args.checkpoint == "all":
        epochs = sorted(args.run_dir.glob("epoch_*.ckpt"))
        ambiguous = [p.name for p in epochs if not p.stem[6:].isdigit()]
        if ambiguous:
            raise ValueError(f"Versioned or ambiguous epoch checkpoints: {ambiguous}. "
                             "Verify the interrupted save and select an explicit --checkpoint epoch_NN.")
        numbers = [int(p.stem[6:]) for p in epochs]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Multiple checkpoint files have the same epoch number.")
        paths = [args.run_dir / "initial.ckpt", *epochs]
    else:
        if args.checkpoint != "initial" and not (
            args.checkpoint.startswith("epoch_") and args.checkpoint[6:].isdigit()
        ):
            raise ValueError("Checkpoint must be all, initial or epoch_NN.")
        paths = [args.run_dir / f"{args.checkpoint}.ckpt"]
    for checkpoint in paths:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = args.run_dir / "embeddings" / checkpoint.stem
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(checkpoint_sha256=sha(checkpoint), roles_sha256=sha(args.roles),
                        config_sha256=sha(args.run_dir / "config.json"),
                        generator_sha256=sha(here / "vendor_bt/generate_embeddings_fold.py"),
                        exporter_sha256=sha(__file__), source_stats=source_stats,
                        batch_size=args.batch_size)
        record = output / "manifest.json"
        if record.exists():
            old = json.loads(record.read_text())
            if all(old.get(k) == v for k, v in manifest.items()) and all(
                (output / name).is_file() and sha(output / name) == value
                for name, value in old["output_sha256"].items()
            ):
                print(f"Verified cached export: {checkpoint.name}", flush=True)
                continue
            raise ValueError(f"Export inputs changed: {output}")
        spec = importlib.util.spec_from_file_location("pilot_export", here / "vendor_bt/generate_embeddings_fold.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            work = Path(temporary)
            roles.to_csv(work / "roles.csv", index=False)
            module.CHECKPOINT_PATH = checkpoint
            module.SAMPLE_ROLES_PATH = work / "roles.csv"
            module.SAMPLE_DIR = args.samples_dir
            module.OUTPUT_DIR = work
            module.RUN_DIR = args.run_dir
            module.SEQUENCE_LENGTH = config["sequence_length"]
            module.INFERENCE_SUBSET_SIZE = config["sample_subset_size"]
            module.RANDOM_SEED = 0
            module.EMBEDDING_BATCH_SIZE = args.batch_size
            names = ["embeddings_preprojector_256D.csv", "embeddings_projector_64D.csv", "embedding_qc.csv"]
            module.PREPROJECTOR_OUTPUT, module.PROJECTOR_OUTPUT, module.QC_OUTPUT = [work / n for n in names]
            module.main()
            for name in names:
                frame = pd.read_csv(work / name)
                if not frame.Sample.is_unique or set(frame.Sample) != set(roles.spygen_code):
                    raise ValueError(f"Missing or duplicate output samples: {name}")
                if not frame.set_index("Sample").role.equals(
                    roles.set_index("spygen_code").role.reindex(frame.Sample).rename_axis("Sample")
                ):
                    raise ValueError(f"Unexpected output roles: {name}")
                frame.to_csv(output / name, index=False)
        if source_stats != {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in samples}:
            raise RuntimeError("Raw inputs changed during export.")
        manifest["output_sha256"] = {name: sha(output / name) for name in names}
        atomic_json(record, manifest)
        print(f"Export complete: {output}", flush=True)


if __name__ == "__main__":
    main()
