"""Reconstruct untrained BT checkpoints using the architecture of existing final runs."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def generate(final_path, output_path, fold, seed):
    import pytorch_lightning as pl
    import torch
    from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder

    config = json.loads(final_path.with_name("config.json").read_text())
    if (config["outer_fold"], config["seed"]) != (fold, seed):
        raise ValueError(f"Fold or seed differs from the training configuration: {final_path}")
    if config.get("sample_repr_dim") != 256:
        raise ValueError(f"Expected the 256D BT encoder: {final_path}")

    # Read only trusted checkpoints from the user's own BT runs.
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    hparams = dict(final["hyper_parameters"])
    pl.seed_everything(seed, workers=True)
    model = SelfAttentionBarlowTwinsEmbedder(**hparams)
    initial_state = model.state_dict()
    trained_state = final["state_dict"]
    if initial_state.keys() != trained_state.keys() or any(
        value.shape != trained_state[name].shape for name, value in initial_state.items()
    ):
        raise ValueError(f"The server BT code does not match the architecture in {final_path}")
    if all(torch.equal(value, trained_state[name]) for name, value in model.named_parameters()):
        raise ValueError(f"Reconstructed parameters are identical to those in {final_path}")
    del final, trained_state

    checkpoint = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
        "state_dict": initial_state,
        "hyper_parameters": hparams,
        "initialization": {
            "kind": "seeded_reconstruction",
            "outer_fold": fold,
            "seed": seed,
            "reference_checkpoint": str(final_path.resolve()),
            "torch_version": str(torch.__version__),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp:
        staged = Path(tmp) / "initial.ckpt"
        torch.save(checkpoint, staged)
        os.link(staged, output_path)  # Publish atomically without replacing an existing file.
    print(f"Fold {fold:02d}, seed {seed}: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("runs_bt_432samples_initial"))
    parser.add_argument("--outer_fold", type=int, choices=range(1, 5), nargs="+", default=list(range(1, 5)))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    jobs = []
    for fold in dict.fromkeys(args.outer_fold):
        relative = Path(f"fold_{fold:02d}") / f"seed_{args.seed}"
        final = args.trained_root / relative / "final.ckpt"
        output = args.output_root / relative / "initial.ckpt"
        for required in (final, final.with_name("config.json")):
            if not required.is_file():
                parser.error(f"Required input missing: {required}")
        if output.exists():
            parser.error(f"Output already exists: {output}. Select a fresh output root or another fold.")
        jobs.append((final, output, fold))

    for final, output, fold in jobs:
        generate(final, output, fold, args.seed)


if __name__ == "__main__":
    main()
