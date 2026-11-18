"""Resume a one-epoch BT run and save checkpoints at total epochs 2 and 5."""
import argparse
import ast
import json
import hashlib
import inspect
import shutil
import sys
from pathlib import Path


def add_checkpoints(source, shuffled=False):
    tree = ast.parse(source)
    if shuffled:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name == "DataLoader":
                for keyword in node.keywords:
                    if keyword.arg == "persistent_workers":
                        keyword.value = ast.Constant(False)
    trainers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr == "Trainer"
    ]
    if len(trainers) != 1:
        raise ValueError("Expected one pl.Trainer call in training_BarlowTwins_kfold.py.")
    trainer = trainers[0]
    if any(keyword.arg == "callbacks" for keyword in trainer.keywords):
        raise ValueError("The upstream trainer already defines callbacks. Review before adding duration checkpoints.")
    if any(keyword.arg == "reload_dataloaders_every_n_epochs" for keyword in trainer.keywords):
        raise ValueError("Review the upstream dataloader reload setting.")
    trainer.keywords.append(ast.keyword(arg="reload_dataloaders_every_n_epochs", value=ast.Constant(1)))
    dataset_calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "BarlowTwinsDataset"
    ]
    if len(dataset_calls) != 1:
        raise ValueError("Expected one BarlowTwinsDataset construction.")
    dataset_calls[0].func.id = "make_duration_dataset"
    trainer.keywords.append(ast.keyword(
        arg="callbacks",
        value=ast.List(elts=[ast.Call(func=ast.Name(id="DurationCheckpoints", ctx=ast.Load()),
                                    args=[], keywords=[])], ctx=ast.Load()),
    ))
    fits = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "trainer"
        and node.func.attr == "fit"
    ]
    if len(fits) != 1 or any(k.arg == "ckpt_path" for k in fits[0].keywords):
        raise ValueError("Expected one trainer.fit call without an existing resume checkpoint.")
    fits[0].keywords.append(ast.keyword(
        arg="ckpt_path", value=ast.Name(id="resume_checkpoint", ctx=ast.Load()),
    ))
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--resume_checkpoint", type=Path, required=True)
    parser.add_argument("--views", choices=["fixed", "shuffled"], required=True)
    args, remaining = parser.parse_known_args()
    resume = args.resume_checkpoint.resolve()
    if not resume.is_file():
        raise FileNotFoundError(resume)
    import pytorch_lightning as pl
    import torch

    checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
    completed = checkpoint.get("loops", {}).get("fit_loop", {}).get(
        "epoch_progress", {}).get("current", {}).get("completed")
    if completed != 1 or checkpoint.get("global_step", 0) <= 0:
        raise ValueError("Expected a full checkpoint saved after exactly one completed epoch.")
    if not checkpoint.get("optimizer_states") or not checkpoint.get("loops"):
        raise ValueError("A full training checkpoint with optimizer and loop state is required.")
    config = json.loads(resume.with_name("config.json").read_text())
    options = argparse.ArgumentParser(add_help=False)
    settings = {
        "outer_fold": int, "seed": int, "sequence_length": int, "sample_subset_size": int,
        "batch_size": int, "token_emb_dim": int, "sample_repr_dim": int, "sample_emb_dim": int,
        "barlow_twins_lambda": float, "initial_learning_rate": float, "weight_decay": float,
    }
    for name, dtype in settings.items():
        options.add_argument("--" + name, type=dtype)
    supplied, _ = options.parse_known_args(remaining)
    for name in settings:
        value = getattr(supplied, name)
        if value is None or config.get(name) != value:
            raise ValueError(f"Resume configuration differs for {name}: {value} vs {config.get(name)}")
    if "split_sha256" in config:
        split_option = argparse.ArgumentParser(add_help=False)
        split_option.add_argument("--split_dir", type=Path, required=True)
        split_args, _ = split_option.parse_known_args(remaining)
        for name, expected in config["split_sha256"].items():
            if hashlib.sha256((split_args.split_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Resume split differs: {name}")
    del checkpoint
    from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset
    import numpy as np
    import pandas as pd

    datasets = []

    def make_duration_dataset(*dataset_args, **dataset_kwargs):
        dataset = BarlowTwinsDataset(*dataset_args, **dataset_kwargs)
        bound = inspect.signature(BarlowTwinsDataset).bind(*dataset_args, **dataset_kwargs)
        bound.apply_defaults()
        datasets.append((dataset, list(dataset.files), dict(bound.arguments)))
        return dataset

    def prepare_views(run_dir, epoch):
        if args.views == "fixed":
            return
        cache = run_dir / "shuffled_reads"
        # Called before workers start or after every batch of the epoch was consumed.
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir()
        manifest = []
        for dataset, original_files, dataset_options in datasets:
            shuffled_files = []
            for source in original_files:
                source = Path(source)
                sample_seed = int.from_bytes(hashlib.sha256(source.stem.encode()).digest()[:4], "little")
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, sample_seed]))
                frame = pd.read_csv(source)
                order = rng.permutation(len(frame))
                output = cache / source.name
                frame.iloc[order].to_csv(output, index=False)
                shuffled_files.append(output)
                manifest.append({"sample": source.stem, "rows": len(frame), "seed": sample_seed})
                del frame, order
            # Rebuild byte ranges, CSV layouts and file stats after rewriting the CSVs.
            options = {**dataset_options, "sample_files": shuffled_files}
            refreshed = BarlowTwinsDataset(**options)
            if [Path(p).name for p in refreshed.files] != [Path(p).name for p in original_files]:
                raise ValueError("Shuffling changed the accepted training samples.")
            if refreshed.accumulated_num_subsets != dataset.accumulated_num_subsets:
                raise ValueError("Shuffling changed the number of chunks per training sample.")
            # DataLoader and its sampler retain this dataset instance.
            dataset.__dict__.clear()
            dataset.__dict__.update(refreshed.__dict__)
        (run_dir / f"shuffle_epoch_{epoch:02d}.json").write_text(json.dumps({
            "epoch": epoch, "seed": config["seed"], "samples": manifest,
            "paired_rows_kept_together": True, "sample_chunk_counts_unchanged": True,
        }, indent=2) + "\n")
        print(f"Prepared shuffled read pairs for total epoch {epoch}", flush=True)
    sys.argv = [str(Path(__file__)), *remaining]

    class DurationCheckpoints(pl.Callback):
        def on_fit_start(self, trainer, pl_module):
            if trainer.max_epochs != 5:
                raise ValueError("This experiment requires --max_epochs 5.")
            path = Path(trainer.default_root_dir) / "epoch_01.ckpt"
            shutil.copy2(resume, path)
            (path.parent / "resume_source.txt").write_text(str(resume) + "\n")
            (path.parent / "duration_protocol.json").write_text(json.dumps({
                "views": args.views, "resume_checkpoint": str(resume),
                "resume_checkpoint_sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
                "seed": config["seed"], "target_total_epochs": 5,
                "dataset_source_sha256": hashlib.sha256(Path(sys.modules[BarlowTwinsDataset.__module__].__file__).read_bytes()).hexdigest(),
            }, indent=2) + "\n")
            prepare_views(path.parent, 2)
            print(f"Resuming from one completed epoch: {resume}, views={args.views}", flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            if epoch in (2, 5):
                path = Path(trainer.default_root_dir) / f"epoch_{epoch:02d}.ckpt"
                trainer.save_checkpoint(str(path))
                print(f"Completed epoch {epoch}, global step {trainer.global_step}: {path}", flush=True)
            if epoch < trainer.max_epochs:
                prepare_views(Path(trainer.default_root_dir), epoch + 1)

        def on_fit_end(self, trainer, pl_module):
            cache = Path(trainer.default_root_dir) / "shuffled_reads"
            if args.views == "shuffled" and cache.exists():
                shutil.rmtree(cache)

    trainer_path = Path(__file__).resolve().with_name("training_BarlowTwins_kfold.py")
    source = trainer_path.read_text()
    tree = add_checkpoints(source, shuffled=args.views == "shuffled")
    namespace = {
        "__name__": "__main__",
        "__file__": str(trainer_path),
        "DurationCheckpoints": DurationCheckpoints,
        "resume_checkpoint": str(resume),
        "make_duration_dataset": make_duration_dataset,
    }
    exec(compile(tree, str(trainer_path), "exec"), namespace)


if __name__ == "__main__":
    main()
