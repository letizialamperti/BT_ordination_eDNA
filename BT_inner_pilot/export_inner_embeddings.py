"""Export fixed 500-pair mean representations for training and inner validation only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import pandas as pd


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
    roles = pd.read_csv(args.roles)
    assert roles.spygen_code.is_unique and set(roles.role) == {"inner_train", "inner_validation"}
    config = json.loads((args.run_dir / "config.json").read_text())
    if args.checkpoint == "all":
        paths = [args.run_dir / "initial.ckpt", *sorted(args.run_dir.glob("epoch_*.ckpt"))]
    else:
        paths = [args.run_dir / f"{args.checkpoint}.ckpt"]
    for checkpoint in paths:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = args.run_dir / "embeddings" / checkpoint.stem
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(checkpoint_sha256=sha(checkpoint), roles_sha256=sha(args.roles),
                        config_sha256=sha(args.run_dir / "config.json"),
                        generator_sha256=sha(here / "vendor_bt/generate_embeddings_fold.py"),
                        batch_size=args.batch_size)
        record = output / "manifest.json"
        if record.exists():
            old = json.loads(record.read_text())
            if all(old.get(k) == v for k, v in manifest.items()) and all(
                sha(output / name) == value for name, value in old["output_sha256"].items()
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
                assert frame.Sample.is_unique and set(frame.Sample) == set(roles.spygen_code)
                frame.to_csv(output / name, index=False)
        manifest["output_sha256"] = {name: sha(output / name) for name in names}
        record.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Export complete: {output}", flush=True)


if __name__ == "__main__":
    main()
