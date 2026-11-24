"""Local inference/validation; shared-storage access uses bounded verified I/O."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import pandas as pd
import export_io as io


def quarantine(directory, output, reason):
    target = output / "_quarantine" / f"{directory.name}_{uuid.uuid4().hex}"
    def move():
        target.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            directory.replace(target)
        elif not target.exists():
            raise FileNotFoundError(directory)
    io.retry(move, f"quarantine {directory}")
    io.write_json(target / "repair_reason.json", dict(reason=str(reason), original=str(directory)))
    print(f"Preserved invalid cached sample: {target}", flush=True)


def run_export(args, api):
    run = args.run_dir
    config = io.read_json(run / "config.json")
    complete = io.read_json(run / "training_complete.json")
    checkpoint = run / "final.ckpt"
    checkpoint_sha = io.retry(lambda: api.sha(checkpoint), f"hash {checkpoint}")
    config_sha = io.retry(lambda: api.sha(run / "config.json"), "hash training config")
    if (complete["completed_epochs"] != 6 or complete["checkpoint_sha256"] != checkpoint_sha
        or complete["config_sha256"] != config_sha):
        raise ValueError("Completed six-epoch training not verified.")
    role_path = run / "sample_roles.csv"
    role_sha = io.retry(lambda: api.sha(role_path), "hash roles")
    if role_sha != config["roles_sha256"]:
        raise ValueError("Frozen outer roles changed.")
    roles = io.retry(lambda: api.read_roles(role_path, config["outer_fold"], config["smoke_test"]), "read roles")
    paths = [args.samples_dir / f"{code}.csv" for code in roles.spygen_code]
    stats = io.retry(lambda: api.file_stats(paths), "stat original sample files")
    training_paths = [args.samples_dir / f"{code}.csv" for code in roles.loc[roles.role.eq("train"), "spygen_code"]]
    if io.retry(lambda: api.file_stats(training_paths), "stat training files") != config["raw_input_stats"]:
        raise ValueError("Training input CSVs changed since fitting.")
    output = run / "embeddings/epoch_06"
    io.retry(lambda: output.mkdir(parents=True, exist_ok=True), "create export directory")
    expected = dict(checkpoint="final.ckpt", checkpoint_sha256=checkpoint_sha,
                    completed_epochs=6, config_sha256=config_sha, roles_sha256=role_sha,
                    source_stats=stats, batch_size=args.batch_size,
                    exporter_sha256=api.sha(api.__file__),
                    generator_sha256=api.sha(api.HERE / "vendor_bt/generate_embeddings_fold.py"),
                    io_source_sha256={"staging":api.sha(__file__), "transfers":api.sha(io.__file__)},
                    io_strategy="node-local-inference-verified-transfer-v1")
    protocol = output / "export_protocol.json"
    if io.retry(protocol.exists, "check export protocol"):
        old = io.read_json(protocol)
        api.check_protocol(old, expected, output)
        if old != expected:
            archive = output / f"export_protocol_before_{old['exporter_sha256'][:12]}.json"
            if not io.retry(archive.exists, "check old protocol archive"):
                io.write_json(archive, old)
    io.write_json(protocol, expected)
    marker = output / "manifest.json"
    if io.retry(marker.exists, "check final manifest"):
        record = io.read_json(marker, manifest=True)
        api.check_protocol(record, expected, output)
        io.retry(lambda: api.verified(output, record, expected), "verify completed export",
                 extra=(api.CachedOutputError,))
        io.write_json(run / "progress.json", dict(phase="complete", exported_samples=len(roles)))
        print(f"Verified completed final embeddings: {output}", flush=True)
        return
    staging = Path(tempfile.mkdtemp(prefix="bt_export_", dir=args.scratch_dir))
    success = False
    print(f"Node-local export staging: {staging}", flush=True)
    try:
        process(args, api, run, roles, config, stats, paths, checkpoint, checkpoint_sha, output, expected, staging)
        success = True
    finally:
        if success:
            shutil.rmtree(staging)
        else:
            print(f"Export interrupted; local staging retained for diagnostics: {staging}", flush=True)


def process(args, api, run, roles, config, stats, paths, checkpoint, checkpoint_sha, output, expected, staging):
    cache = output / "_samples"
    io.retry(lambda: cache.mkdir(exist_ok=True), "create sample cache")
    local_cache = staging / "samples";local_cache.mkdir()
    available = {}
    # Audit ALL existing entries before loading a GPU model. A repeated transport
    # error is not called corruption. Only stable integrity/schema failures are
    # quarantined after bounded verification attempts.
    io.write_json(run / "progress.json", dict(phase="checking_cache", total_samples=len(roles)))
    for position, row in enumerate(roles.itertuples(index=False), 1):
        directory = cache / row.spygen_code;record = directory / "manifest.json"
        if not io.retry(record.exists, f"check cache manifest {row.spygen_code}"):
            continue
        destination = local_cache / row.spygen_code
        sample_input = dict(Sample=row.spygen_code, role=row.role,
                            source_stat=stats[str((args.samples_dir / f"{row.spygen_code}.csv").resolve())])
        try:
            m = io.read_json(record, manifest=True)
            api.check_protocol(m, sample_input, directory)
            if set(m.get("output_sha256",{})) != set(api.NAMES):
                raise api.CachedOutputError("Incomplete sample manifest")
            for name,digest in m["output_sha256"].items():
                io.copy_verified(directory/name, destination/name, expected_sha=digest)
            api.validate_outputs(destination, row.spygen_code, row.role)
        except (ValueError, FloatingPointError, FileNotFoundError) as exc:
            quarantine(directory, output, exc)
            if destination.exists():shutil.rmtree(destination)
            continue
        available[row.spygen_code] = destination
        print(f"[{position}/{len(roles)}] Verified cached sample: {row.spygen_code}", flush=True)
    io.write_json(run / "progress.json", dict(phase="export", completed_samples=len(available),
                                              total_samples=len(roles), cache_audit_complete=True))
    print(f"Cache audit complete: {len(available)}/{len(roles)} samples verified locally", flush=True)
    missing = [code for code in roles.spygen_code if code not in available]
    if missing:
        # Only one raw sample is staged at a time, plus the frozen checkpoint.
        sizes = {Path(name).stem:value[0] for name,value in stats.items()}
        required = int(1.2*(io.retry(lambda: checkpoint.stat().st_size,"stat checkpoint") + max(sizes[c] for c in missing)))
        if shutil.disk_usage(staging).free < required:
            raise OSError(f"Insufficient local scratch for checkpoint + largest sample: need about {required/2**30:.1f} GiB")
        local_checkpoint = staging / "final.ckpt"
        io.copy_verified(checkpoint, local_checkpoint, expected_sha=checkpoint_sha)
        spec = importlib.util.spec_from_file_location("frozen_export", api.HERE/"vendor_bt/generate_embeddings_fold.py")
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.CHECKPOINT_PATH=local_checkpoint;module.RUN_DIR=run
        module.SEQUENCE_LENGTH=config["sequence_length"];module.INFERENCE_SUBSET_SIZE=config["sample_subset_size"]
        module.RANDOM_SEED=0;module.EMBEDDING_BATCH_SIZE=args.batch_size
        model=module.SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(local_checkpoint)
        raw_dir=staging/"reads";raw_dir.mkdir();module.SAMPLE_DIR=raw_dir
        for position,row in enumerate(roles.itertuples(index=False),1):
            if row.spygen_code in available:
                continue
            source=args.samples_dir/f"{row.spygen_code}.csv";local_raw=raw_dir/source.name
            if io.retry(lambda: api.file_stats([source]), f"stat {source}") != {str(source.resolve()):stats[str(source.resolve())]}:
                raise ValueError(f"Raw sample changed since export started: {source}")
            io.copy_verified(source,local_raw)
            directory=cache/row.spygen_code
            io.retry(lambda: directory.mkdir(parents=True,exist_ok=True), "create sample destination")
            work=local_cache/row.spygen_code;work.mkdir(exist_ok=True)
            pd.DataFrame([row._asdict()]).to_csv(work/"roles.csv",index=False)
            module.SAMPLE_ROLES_PATH=work/"roles.csv";module.OUTPUT_DIR=work
            module.PREPROJECTOR_OUTPUT,module.PROJECTOR_OUTPUT,module.QC_OUTPUT=[work/n for n in api.NAMES]
            print(f"[{position}/{len(roles)}] Exporting locally {row.spygen_code} ({row.role})", flush=True)
            try:
                module.main(model=model)
                api.validate_outputs(work,row.spygen_code,row.role)
            except Exception as exc:
                diagnostic=api.preserve_failure(work,staging/"failed"/row.spygen_code,row.spygen_code,row.role,exc)
                try:
                    remote=directory/"_diagnostics"/diagnostic.name
                    for p in diagnostic.iterdir():io.copy_verified(p,remote/p.name,expected_sha=api.sha(p))
                except Exception as copy_error:print(f"Diagnostic copy-back failed: {copy_error}",flush=True)
                raise RuntimeError(f"Local inference failed for {row.spygen_code}; preserved at {diagnostic}: {exc}") from exc
            hashes={n:api.sha(work/n) for n in api.NAMES}
            for name in api.NAMES:io.copy_verified(work/name,directory/name,expected_sha=hashes[name])
            if io.retry(lambda: api.file_stats([source]),f"recheck source {source}") != {str(source.resolve()):stats[str(source.resolve())]}:
                raise RuntimeError(f"Raw source changed during inference: {source}")
            io.write_json(directory/"manifest.json",dict(Sample=row.spygen_code,role=row.role,
                          source_stat=stats[str(source.resolve())],output_sha256=hashes))
            available[row.spygen_code]=work
            io.write_json(run/"progress.json",dict(phase="export",completed_samples=len(available),
                                                   total_samples=len(roles),last_sample=row.spygen_code))
            local_raw.unlink()
    if io.retry(lambda: api.file_stats(paths),"verify all original source stats") != stats:
        raise RuntimeError("Input CSVs changed during export")
    combined=staging/"combined";combined.mkdir()
    for name in api.NAMES:
        frame=pd.concat([pd.read_csv(available[code]/name) for code in roles.spygen_code],ignore_index=True)
        if not frame.Sample.is_unique or frame.Sample.tolist()!=roles.spygen_code.tolist() or frame.role.tolist()!=roles.role.tolist():
            raise ValueError(f"Invalid combined sample alignment: {name}")
        frame.to_csv(combined/name,index=False)
        io.copy_verified(combined/name,output/name,expected_sha=api.sha(combined/name))
    io.write_json(output/"manifest.json",dict(expected,n_samples=len(roles),role_counts=roles.role.value_counts().to_dict(),
                                             output_sha256={n:api.sha(combined/n) for n in api.NAMES}))
    io.write_json(run/"progress.json",dict(phase="complete",exported_samples=len(roles)))
    print(f"EXPORT COMPLETE: {output}",flush=True)
