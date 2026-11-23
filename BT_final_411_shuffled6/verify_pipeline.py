"""CPU regression test with the real imported BT model and synthetic reads only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from final_checks import sha, read_roles


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--report", type=Path, default=HERE / "verification.json")
    args = parser.parse_args()
    checks = []
    cohorts, counts = [], []
    for fold in [1,2,3,4]:
        r = read_roles(HERE / f"splits/fold_{fold:02d}.csv", fold)
        cohorts.append(set(r.spygen_code));counts.append(r.role.eq("outer_eval").sum())
    assert all(c == cohorts[0] for c in cohorts) and counts == [95,107,122,87]
    checks.append("All four outer splits share 411 IDs; outer counts 95/107/122/87")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "bt-final-test-mpl"))
    with tempfile.TemporaryDirectory(prefix="bt_final_test_") as temporary:
        root = Path(temporary);samples = root / "reads";samples.mkdir()
        records = [dict(spygen_code=f"sample_{block}_{j}", spatial_block=block,
                        role="outer_eval" if block == 1 else "train")
                   for block in [1,2,3,4] for j in range(2)]
        roles = pd.DataFrame(records);roles.to_csv(root / "roles.csv",index=False)
        def write_sample(code):
            rows = []
            for i in range(17):
                seq = "".join("ACGT"[(i // (4**j)) % 4] for j in [3,2,1,0])
                rows.append(dict(Forward=" "+seq+" ", Reverse=seq[::-1], Extra="x"*(i+1)))
            rows.append(dict(Forward="none", Reverse="ACGT", Extra="missing"))
            pd.DataFrame(rows).to_csv(samples / f"{code}.csv",index=False)
        for row in records:
            if row["role"] == "train":write_sample(row["spygen_code"])
        def command(script, extra, success=True):
            r = subprocess.run([sys.executable,str(HERE/script),*map(str,extra)],env=env,capture_output=True,text=True)
            if (r.returncode == 0) != success:
                raise RuntimeError(f"{script}: exit {r.returncode}\n{r.stdout}\n{r.stderr}")
            return r
        train_args = ["--fold",1,"--samples-dir",samples,"--output-root",root/"runs",
                      "--roles",root/"roles.csv","--scratch-dir",root,"--workers",args.workers,
                      "--accelerator","cpu","--smoke-test"]
        command("train_final.py",train_args)
        run = root / "runs/fold_01/seed_0"
        assert all((run/f"epoch_{i:02d}.ckpt").is_file() for i in range(1,7))
        final = torch.load(run/"final.ckpt",map_location="cpu",weights_only=False)
        assert final["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 6
        for p in (run/"metrics").glob("version_*/metrics.csv"):
            assert not any(c.startswith("val_") for c in pd.read_csv(p).columns)
        assert len(pd.read_csv(run/"train_qc.csv"))==6
        checks.append("Real BT model trains exactly six shuffled epochs without opening absent outer CSVs or validating")
        checkpoint_hash = sha(run/"final.ckpt")
        skipped = command("train_final.py",train_args)
        assert "Verified completed training" in skipped.stdout and sha(run/"final.ckpt")==checkpoint_hash
        checks.append("Completed training is verified and skipped on pipeline relaunch")
        # Resume the last training epoch using a preserved full checkpoint. The
        # previous epoch_06 file exists: it must not create an epoch_06-v1 file.
        import shutil
        shutil.copy2(run/"epoch_05.ckpt",run/"last.ckpt")
        (run/"training_complete.json").unlink()
        command("train_final.py",train_args)
        resumed = torch.load(run/"final.ckpt",map_location="cpu",weights_only=False)
        assert all(torch.equal(v,resumed["state_dict"][k]) for k,v in final["state_dict"].items())
        assert not list(run.glob("epoch_*-v*.ckpt"))
        checks.append("Resume from epoch 5 reproduces final weights exactly and avoids versioned epoch duplicates")
        for row in records:
            if row["role"]=="outer_eval":write_sample(row["spygen_code"])
        export_args = ["--run-dir",run,"--samples-dir",samples,"--batch-size",3]
        command("export_final.py",export_args)
        output = run/"embeddings/epoch_06"
        assert len(pd.read_csv(output/"embeddings_preprojector_256D.csv"))==8
        assert len(list((output/"_samples").glob("*/manifest.json")))==8
        checks.append("Final export includes train and held-out samples, with 256D/64D outputs and per-sample manifests")
        # Simulate export termination after all sample transactions but before
        # the combined CSV transaction. Reuse must not rewrite cached samples.
        sample_hashes = {str(p.relative_to(output)):sha(p) for p in (output/"_samples").glob("*/manifest.json")}
        (output/"manifest.json").unlink()
        recovered = command("export_final.py",export_args)
        assert recovered.stdout.count("Verified cached sample")==8
        assert sample_hashes=={str(p.relative_to(output)):sha(p) for p in (output/"_samples").glob("*/manifest.json")}
        skipped = command("export_final.py",export_args)
        assert "Verified completed final embeddings" in skipped.stdout
        checks.append("Interrupted export reuses sample transactions; complete exports are hash-verified and skipped")
        # Compare the incremental export against the original single-pass math.
        spec=importlib.util.spec_from_file_location("reference_generator",HERE/"vendor_bt/generate_embeddings_fold.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ref=root/"reference";ref.mkdir()
        module.CHECKPOINT_PATH=run/"final.ckpt";module.SAMPLE_ROLES_PATH=run/"sample_roles.csv"
        module.SAMPLE_DIR=samples;module.RUN_DIR=run;module.OUTPUT_DIR=ref
        module.SEQUENCE_LENGTH=8;module.INFERENCE_SUBSET_SIZE=2;module.EMBEDDING_BATCH_SIZE=3
        module.PREPROJECTOR_OUTPUT=ref/"embeddings_preprojector_256D.csv"
        module.PROJECTOR_OUTPUT=ref/"embeddings_projector_64D.csv";module.QC_OUTPUT=ref/"embedding_qc.csv"
        module.main()
        for name in ["embeddings_preprojector_256D.csv","embeddings_projector_64D.csv"]:
            a=pd.read_csv(output/name);b=pd.read_csv(ref/name)
            assert a[["Sample","role"]].equals(b[["Sample","role"]])
            assert np.allclose(a.select_dtypes("number"),b.select_dtypes("number"),rtol=1e-6,atol=1e-7)
        checks.append("Incremental export matches original single-pass embeddings for every sample and both layers")
        # Altering an original training CSV must stop the pipeline before reuse.
        changed=samples/"sample_2_0.csv";changed.write_text(changed.read_text()+"\n")
        rejected=command("train_final.py",train_args,success=False)
        assert "Run inputs, recipe or code changed" in rejected.stderr
        checks.append("Changed raw training input is rejected on relaunch")
    report=dict(status="PASS",workers=args.workers,checks=checks,
                source_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob("*.py")})
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
