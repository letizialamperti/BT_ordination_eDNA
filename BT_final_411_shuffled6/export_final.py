"""Export the frozen six-epoch model for train and outer IDs, resuming per sample."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from final_checks import sha, read_roles, file_stats, run_lock, atomic_json


HERE = Path(__file__).resolve().parent
NAMES = ["embeddings_preprojector_256D.csv", "embeddings_projector_64D.csv", "embedding_qc.csv"]


def verified(directory, record, expected=None):
    if expected is not None and any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Export protocol or input changed: {directory}")
    if set(record.get("output_sha256", {})) != set(NAMES):
        raise ValueError(f"Incomplete export manifest: {directory}")
    for name, digest in record["output_sha256"].items():
        if not (directory / name).is_file() or sha(directory / name) != digest:
            raise ValueError(f"Changed cached export: {directory / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    with run_lock(args.run_dir):
        export(args)


def export(args):
    run = args.run_dir
    config = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    checkpoint = run / "final.ckpt"
    checkpoint_sha = sha(checkpoint)
    if (complete["completed_epochs"] != 6 or complete["checkpoint_sha256"] != checkpoint_sha
        or complete["config_sha256"] != sha(run / "config.json")):
        raise ValueError("Completed six-epoch training not verified.")
    role_path = run / "sample_roles.csv"
    if sha(role_path) != config["roles_sha256"]:
        raise ValueError("Frozen outer roles changed.")
    roles = read_roles(role_path, config["outer_fold"], config["smoke_test"])
    paths = [args.samples_dir / f"{code}.csv" for code in roles.spygen_code]
    stats = file_stats(paths)
    training_paths = [args.samples_dir / f"{code}.csv"
                      for code in roles.loc[roles.role.eq("train"), "spygen_code"]]
    if file_stats(training_paths) != config["raw_input_stats"]:
        raise ValueError("Training input CSVs changed since fitting.")
    output = run / "embeddings/epoch_06"
    output.mkdir(parents=True, exist_ok=True)
    expected = dict(checkpoint="final.ckpt", checkpoint_sha256=checkpoint_sha,
                    completed_epochs=6, config_sha256=sha(run / "config.json"),
                    roles_sha256=sha(role_path), source_stats=stats, batch_size=args.batch_size,
                    exporter_sha256=sha(__file__), generator_sha256=sha(HERE / "vendor_bt/generate_embeddings_fold.py"))
    marker = output / "manifest.json"
    if marker.exists():
        verified(output, json.loads(marker.read_text()), expected)
        atomic_json(run / "progress.json", dict(phase="complete", exported_samples=len(roles)))
        print(f"Verified completed final embeddings: {output}", flush=True)
        return
    protocol = output / "export_protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != expected:
        raise ValueError("Partial export inputs or code changed; preserve and inspect this output.")
    atomic_json(protocol, expected)
    atomic_json(run / "progress.json", dict(phase="initializing_export", total_samples=len(roles)))
    spec = importlib.util.spec_from_file_location("frozen_export", HERE / "vendor_bt/generate_embeddings_fold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CHECKPOINT_PATH = checkpoint
    module.SAMPLE_DIR = args.samples_dir
    module.RUN_DIR = run
    module.SEQUENCE_LENGTH = config["sequence_length"]
    module.INFERENCE_SUBSET_SIZE = config["sample_subset_size"]
    module.RANDOM_SEED = 0
    module.EMBEDDING_BATCH_SIZE = args.batch_size
    # Load once for all samples. Inference/pooling and per-sample random_state=0
    # remain identical to the original generator.
    model = module.SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(checkpoint)
    cache = output / "_samples"
    cache.mkdir(exist_ok=True)
    for position, row in enumerate(roles.itertuples(index=False), 1):
        directory = cache / row.spygen_code
        directory.mkdir(exist_ok=True)
        record = directory / "manifest.json"
        sample_input = dict(Sample=row.spygen_code, role=row.role,
                            source_stat=stats[str((args.samples_dir / f"{row.spygen_code}.csv").resolve())])
        if record.exists():
            verified(directory, json.loads(record.read_text()), sample_input)
            print(f"[{position}/{len(roles)}] Verified cached sample: {row.spygen_code}", flush=True)
        else:
            print(f"[{position}/{len(roles)}] Exporting {row.spygen_code} ({row.role})", flush=True)
            with tempfile.TemporaryDirectory(prefix="work_", dir=directory) as temporary:
                work = Path(temporary)
                pd.DataFrame([row._asdict()]).to_csv(work / "roles.csv", index=False)
                module.SAMPLE_ROLES_PATH = work / "roles.csv"
                module.OUTPUT_DIR = work
                module.PREPROJECTOR_OUTPUT, module.PROJECTOR_OUTPUT, module.QC_OUTPUT = [work / n for n in NAMES]
                module.main(model=model)
                for name in NAMES:
                    frame = pd.read_csv(work / name)
                    if len(frame) != 1 or frame.Sample.iloc[0] != row.spygen_code or frame.role.iloc[0] != row.role:
                        raise ValueError(f"Incomplete inference subset or invalid output: {row.spygen_code}/{name}")
                    if name != "embedding_qc.csv" and not np.isfinite(frame.filter(regex=r"^(Dim|Std)").to_numpy()).all():
                        raise FloatingPointError(f"Non-finite embedding: {row.spygen_code}")
                    (work / name).replace(directory / name)
            if file_stats([args.samples_dir / f"{row.spygen_code}.csv"])[str((args.samples_dir / f"{row.spygen_code}.csv").resolve())] != sample_input["source_stat"]:
                raise RuntimeError(f"Input changed during export: {row.spygen_code}")
            atomic_json(record, dict(sample_input, output_sha256={n: sha(directory / n) for n in NAMES}))
        atomic_json(run / "progress.json", dict(phase="export", completed_samples=position,
                                                total_samples=len(roles), last_sample=row.spygen_code))
    if file_stats(paths) != stats:
        raise RuntimeError("Input CSVs changed during export.")
    for name in NAMES:
        frame = pd.concat([pd.read_csv(cache / code / name) for code in roles.spygen_code], ignore_index=True)
        if not frame.Sample.is_unique or frame.Sample.tolist() != roles.spygen_code.tolist() or frame.role.tolist() != roles.role.tolist():
            raise ValueError(f"Invalid final sample alignment: {name}")
        temporary = output / (name + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output / name)
    atomic_json(marker, dict(expected, n_samples=len(roles), role_counts=roles.role.value_counts().to_dict(),
                             output_sha256={n: sha(output / n) for n in NAMES}))
    atomic_json(run / "progress.json", dict(phase="complete", exported_samples=len(roles)))
    print(f"EXPORT COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
