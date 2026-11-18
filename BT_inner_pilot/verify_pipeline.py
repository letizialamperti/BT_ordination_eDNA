"""Synthetic CPU regression checks; never opens real study sample CSVs.

Run with the BT repository on PYTHONPATH, using the zioboia environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from pilot_checks import read_roles, run_lock, atomic_json
from train_bt_inner_pilot import CompleteBatches


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=HERE / "audit_report.json")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    checks = []
    for n in range(2, 100):
        for size in range(2, 20):
            batcher = CompleteBatches(list(range(n)), size)
            batches = list(batcher)
            assert len(batches) == len(batcher)
            assert all(2 <= len(b) <= size + 1 for b in batches)
            assert sum(batches, []) == list(range(n))
    checks.append("1764 batch-size/length combinations retain every chunk with no singleton")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "bt-audit-mpl"))
    with tempfile.TemporaryDirectory(prefix="bt_pipeline_audit_") as temporary:
        root = Path(temporary)

        def run(script, arguments, success=True):
            result = subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)],
                                    env=env, capture_output=True, text=True)
            if (result.returncode == 0) != success:
                raise RuntimeError(f"{script} returned {result.returncode}\n{result.stdout}\n{result.stderr}")
            return result

        samples = root / "samples"
        samples.mkdir()
        roles = pd.DataFrame([
            ("train_a", 3, "inner_train"), ("train_b", 4, "inner_train"),
            ("validation", 2, "inner_validation"), ("OUTER_MUST_NOT_BE_OPENED", 1, "outer_eval"),
        ], columns=["spygen_code", "spatial_block", "role"])
        role_path = root / "roles.csv"
        roles.to_csv(role_path, index=False)
        read_roles(role_path)
        for column, value in [("spatial_block", 3), ("spygen_code", "train_a"),
                              ("spygen_code", "../outside")]:
            bad = roles.copy()
            bad.loc[2, column] = value
            bad.to_csv(root / "bad_roles.csv", index=False)
            try:
                read_roles(root / "bad_roles.csv")
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid {column} accepted")
        checks.append("duplicate IDs, unsafe paths and overlapping spatial roles rejected")
        with run_lock(root / "locked"):
            try:
                with run_lock(root / "locked"):
                    raise AssertionError("Concurrent run was accepted")
            except RuntimeError:
                pass
        checks.append("exclusive run lock rejects a competing writer")
        for j, code in enumerate(["train_a", "train_b", "validation"]):
            rows = []
            for i in range(17):
                seq = "".join("ACGT"[((i + j * 3) // (4 ** k)) % 4] for k in [3, 2, 1, 0])
                rows.append(dict(Forward=" " + seq + " ", Reverse=seq[::-1], Extra="x" * (i + 1)))
            rows.extend([dict(Forward="none", Reverse="ACGT", Extra="missing"),
                         dict(Forward="ACGT", Reverse="", Extra="missing")])
            pd.DataFrame(rows).to_csv(samples / f"{code}.csv", index=False)
        common = ["--roles", role_path, "--samples_dir", samples, "--sequence_length", 8,
                  "--sample_subset_size", 2, "--batch_size", 3, "--token_emb_dim", 2,
                  "--validation_chunks_per_sample", 2, "--num_workers", args.workers,
                  "--accelerator", "cpu", "--scratch_dir", root]
        run("train_bt_inner_pilot.py", common + ["--views", "fixed", "--output_root", root / "preflight", "--check-only"])
        checks.append("preflight indexes both roles and exercises indexed views without outer CSV")
        run_dirs = {}
        for mode in ["fixed", "shuffled"]:
            for epochs in [1, 2]:
                run("train_bt_inner_pilot.py", common + ["--views", mode, "--output_root", root / "resumed", "--max_epochs", epochs])
            run("train_bt_inner_pilot.py", common + ["--views", mode, "--output_root", root / "continuous", "--max_epochs", 2])
            resumed = root / "resumed/outer_01_inner_02" / mode / "seed_0"
            continuous = root / "continuous/outer_01_inner_02" / mode / "seed_0"
            run_dirs[mode] = resumed
            a = torch.load(resumed / "final.ckpt", map_location="cpu", weights_only=False)
            b = torch.load(continuous / "final.ckpt", map_location="cpu", weights_only=False)
            for key in a["state_dict"]:
                assert torch.equal(a["state_dict"][key], b["state_dict"][key]), (mode, key)
            assert a["global_step"] == b["global_step"]
            assert a["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 2
            frames = [pd.read_csv(p) for p in (resumed / "metrics").glob("version_*/metrics.csv")]
            metrics = pd.concat(frames)
            assert metrics.val_barlow_loss.notna().sum() == 2
            assert np.isfinite(metrics.val_barlow_loss.dropna()).all()
            assert all((resumed / name).is_file() for name in ["initial.ckpt", "epoch_01.ckpt", "epoch_02.ckpt", "last.ckpt"])
            checks.append(f"{mode}: real BT model trains/validates/saves; resumed weights equal uninterrupted weights exactly")
        initial = [torch.load(run_dirs[m] / "initial.ckpt", map_location="cpu", weights_only=False)["state_dict"]
                   for m in ["fixed", "shuffled"]]
        assert all(torch.equal(v, initial[1][k]) for k, v in initial[0].items())
        assert not list(root.glob("bt_inner_*"))
        checks.append("fixed/shuffled initial weights identical; temporary shuffled CSVs cleaned")
        export_roles = root / "classifier_roles.csv"
        roles.loc[roles.role.ne("outer_eval")].to_csv(export_roles, index=False)
        export_args = ["--run-dir", run_dirs["fixed"], "--samples-dir", samples,
                       "--roles", export_roles, "--batch-size", 3]
        run("export_inner_embeddings.py", export_args)
        cached = run("export_inner_embeddings.py", export_args)
        assert cached.stdout.count("Verified cached export") == 3
        for name in ["initial", "epoch_01", "epoch_02"]:
            frame = pd.read_csv(run_dirs["fixed"] / "embeddings" / name / "embeddings_preprojector_256D.csv")
            assert len(frame) == 3 and np.isfinite(frame[[f"Dim{i}" for i in range(1, 257)]]).all().all()
        checks.append("real frozen export: 256D/64D outputs, sample roles, manifests and cached re-export")
        versioned = run_dirs["fixed"] / "epoch_01-v1.ckpt"
        versioned.write_bytes(b"ambiguous checkpoint sentinel")
        rejected = run("export_inner_embeddings.py", export_args, success=False)
        assert "ambiguous epoch checkpoints" in rejected.stderr
        versioned.unlink()
        checks.append("versioned epoch checkpoints rejected instead of silently scored twice")
        bad = roles.loc[roles.role.ne("outer_eval")].copy()
        bad.loc[0, "spygen_code"] = "OUTER_MUST_NOT_BE_OPENED"
        bad.to_csv(root / "bad_export.csv", index=False)
        run("export_inner_embeddings.py", ["--run-dir", run_dirs["fixed"], "--samples-dir", samples,
            "--roles", root / "bad_export.csv"], success=False)
        checks.append("export rejects an outer sample relabelled as training")
        # Separate synthetic cohort gives every target all classes in both roles.
        cohort, meta = [], []
        rng = np.random.default_rng(0)
        for role, blocks in [("inner_train", [3, 4]), ("inner_validation", [2])]:
            for block in blocks:
                for label in [1, 2, 3, 5]:
                    code = f"sample_{block}_{label}"
                    cohort.append(dict(spygen_code=code, spatial_block=block, role=role))
                    meta.append(dict(spygen_code=code, Index_MPA_num=label))
        pd.DataFrame(cohort).to_csv(root / "class_roles.csv", index=False)
        pd.DataFrame(meta).to_csv(root / "metadata.csv", index=False, sep=";")
        for epoch in ["initial", "epoch_01"]:
            directory = root / "classifier_embeddings" / epoch
            directory.mkdir(parents=True)
            frame = pd.DataFrame(rng.normal(size=(len(cohort), 256)), columns=[f"Dim{i}" for i in range(1,257)])
            frame.insert(0, "Sample", [r["spygen_code"] for r in cohort])
            frame.insert(1, "role", [r["role"] for r in cohort])
            frame.to_csv(directory / "embeddings_preprojector_256D.csv", index=False)
        run("evaluate_inner_classification.py", ["--embeddings-root", root / "classifier_embeddings",
            "--metadata", root / "metadata.csv", "--roles", root / "class_roles.csv",
            "--output", root / "scores", "--classifier-epochs", 1])
        scores = pd.read_csv(root / "scores/inner_scores_by_seed.csv")
        assert len(scores) == 18 and set(scores.n_train) == {8} and set(scores.n_validation) == {4}
        checks.append("classifier: 2 checkpoints x 3 targets x 3 seeds, frozen sample roles")
    atomic_json(args.report, dict(status="PASS", checks=checks, workers=args.workers,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in HERE.glob("*.py")}, torch=torch.__version__))
    print(json.dumps(dict(status="PASS", checks=checks, report=str(args.report)), indent=2))


if __name__ == "__main__":
    main()
