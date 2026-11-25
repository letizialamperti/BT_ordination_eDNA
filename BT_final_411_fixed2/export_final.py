"""Export the fixed two-epoch model for train and outer IDs, resuming per sample."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import sys
import uuid

import numpy as np
import pandas as pd
from final_checks import sha, read_roles, file_stats, run_lock, atomic_json


HERE = Path(__file__).resolve().parent
NAMES = ["embeddings_preprojector_256D.csv", "embeddings_projector_64D.csv", "embedding_qc.csv"]
COMPATIBLE_EXPORTERS = set()


class CachedOutputError(ValueError):
    pass


def check_protocol(record, expected, directory):
    differences = {k for k, value in expected.items() if record.get(k) != value}
    if not differences:
        return
    # Validation/diagnostic-only update: model, generator, data, pooling, batch,
    # checkpoint and source statistics must ALL be unchanged.
    if differences.issubset({"exporter_sha256", "io_source_sha256", "io_strategy"}) and record.get("exporter_sha256") in COMPATIBLE_EXPORTERS:
        print(f"Compatible export I/O update: {directory}; model and numerical recipe unchanged", flush=True)
        return
    raise ValueError(f"Export protocol or input changed: {directory}; fields={sorted(differences)}")


def verified(directory, record, expected=None):
    if expected is not None:
        check_protocol(record, expected, directory)
    if set(record.get("output_sha256", {})) != set(NAMES):
        raise CachedOutputError(f"Incomplete export manifest: {directory}")
    for name, digest in record["output_sha256"].items():
        if not (directory / name).is_file() or sha(directory / name) != digest:
            raise CachedOutputError(f"Changed cached export: {directory / name}")


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
    parser.add_argument("--scratch-dir", type=Path, default=os.environ.get("BT_SHUFFLE_SCRATCH"),
                        help="Node-local staging (the OAR launchers use /var/tmp)")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    with run_lock(args.run_dir):
        export(args)


def export(args):
    from export_staged import run_export
    return run_export(args, sys.modules[__name__])


if __name__ == "__main__":
    main()
