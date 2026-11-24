"""Export the frozen six-epoch model for train and outer IDs, resuming per sample."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
from final_checks import sha, read_roles, file_stats, run_lock, atomic_json


HERE = Path(__file__).resolve().parent
NAMES = ["embeddings_preprojector_256D.csv", "embeddings_projector_64D.csv", "embedding_qc.csv"]
COMPATIBLE_EXPORTERS = {"d991a0cb56a9ba1991bf20056b18ec541a306e6a61c6947f096f9277dadbd1ba"}


def check_protocol(record, expected, directory):
    differences = {k for k, value in expected.items() if record.get(k) != value}
    if not differences:
        return
    # Validation/diagnostic-only update: model, generator, data, pooling, batch,
    # checkpoint and source statistics must ALL be unchanged.
    if differences == {"exporter_sha256"} and record.get("exporter_sha256") in COMPATIBLE_EXPORTERS:
        print(f"Compatible validation-only exporter update: {directory}; cached values remain unchanged", flush=True)
        return
    raise ValueError(f"Export protocol or input changed: {directory}; fields={sorted(differences)}")


def verified(directory, record, expected=None):
    if expected is not None:
        check_protocol(record, expected, directory)
    if set(record.get("output_sha256", {})) != set(NAMES):
        raise ValueError(f"Incomplete export manifest: {directory}")
    for name, digest in record["output_sha256"].items():
        if not (directory / name).is_file() or sha(directory / name) != digest:
            raise ValueError(f"Changed cached export: {directory / name}")


def validate_outputs(work, sample, role):
    """Validate the whole sample before publishing any of its three CSVs."""
    for name in NAMES:
        path = work / name
        try:
            frame = pd.read_csv(path, dtype={"Sample": str, "role": str})
        except Exception as exc:
            raise ValueError(f"{sample}/{name}: CSV cannot be read ({type(exc).__name__}: {exc})") from exc
        actual = {c: frame[c].iloc[0] if len(frame) else None for c in ["Sample", "role"] if c in frame}
        if len(frame) != 1 or actual.get("Sample") != sample or actual.get("role") != role:
            raise ValueError(f"{sample}/{name}: expected one row with Sample={sample!r}, role={role!r}; "
                             f"got rows={len(frame)}, columns={len(frame.columns)}, identifiers={actual!r}")
        if name != "embedding_qc.csv":
            size = 256 if "256D" in name else 64
            columns = [f"Dim{i}" for i in range(1, size+1)] + [f"Standard_dev_{i}" for i in range(1, size+1)]
            missing = [c for c in columns if c not in frame]
            if missing or len(frame.columns) != 2 + 2*size:
                raise ValueError(f"{sample}/{name}: missing columns {missing[:8]}; actual column count={len(frame.columns)}")
            if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
                raise FloatingPointError(f"{sample}/{name}: non-finite mean or standard deviation")
        elif frame.embedding_status.iloc[0] != "ok" or frame.full_subsets.iloc[0] < 1:
            raise ValueError(f"{sample}/{name}: no complete usable inference subset")


def preserve_failure(work, directory, sample, role, error):
    destination = directory / "_diagnostics" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    information = dict(Sample=sample, role=role, error_type=type(error).__name__, error=str(error), files={})
    for name in NAMES:
        source = work / name
        if not source.exists():
            information["files"][name] = {"exists": False}
            continue
        shutil.copy2(source, destination / name)
        item = dict(exists=True, bytes=source.stat().st_size, sha256=sha(source))
        try:
            with source.open(newline="") as stream:
                rows = list(csv.reader(stream))
            item.update(csv_record_count=len(rows), field_counts=[len(r) for r in rows[:3]],
                        first_data_identifiers=rows[1][:2] if len(rows)>1 else [])
        except Exception as exc:
            item["read_error"] = str(exc)
        information["files"][name] = item
    atomic_json(destination / "failure.json", information)
    print(f"Preserved failed sample diagnostics: {destination}", flush=True)
    return destination


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
    if protocol.exists():
        old_protocol = json.loads(protocol.read_text())
        check_protocol(old_protocol, expected, output)
        if old_protocol != expected:
            archive = output / f"export_protocol_before_{old_protocol['exporter_sha256'][:12]}.json"
            if not archive.exists():
                atomic_json(archive, old_protocol)
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
                try:
                    module.main(model=model)
                    validate_outputs(work, row.spygen_code, row.role)
                except Exception as exc:
                    diagnostic = "unavailable"
                    try:
                        diagnostic = preserve_failure(work, directory, row.spygen_code, row.role, exc)
                    except Exception as preservation_error:
                        print(f"Could not preserve diagnostics: {preservation_error}", flush=True)
                    raise RuntimeError(f"Export failed for {row.spygen_code}: {exc}. Diagnostics: {diagnostic}") from exc
                for name in NAMES:
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
