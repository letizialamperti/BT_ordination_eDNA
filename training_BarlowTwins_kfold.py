"""Fixed-duration Barlow Twins training on each outer fold training set only."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_folds(split_dir):
    paths = sorted(Path(split_dir).glob('new_no_coral_split_4_fold_*.csv'))
    if len(paths) != 4:
        raise ValueError('Expected exactly four fold CSVs.')
    folds = []
    for path in paths:
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {'spygen_code', 'set'} <= rows[0].keys():
            raise ValueError(f'Invalid columns: {path}')
        codes = [r['spygen_code'] for r in rows]
        if len(codes) != len(set(codes)) or any(not c or Path(c).name != c for c in codes):
            raise ValueError(f'Duplicate or invalid sample IDs: {path}')
        if set(r['set'] for r in rows) != {'train', 'validation'}:
            raise ValueError(f'Expected train and validation only: {path}')
        folds.append({r['spygen_code']: r['set'] for r in rows})
    universe = set(folds[0])
    if any(set(f) != universe for f in folds):
        raise ValueError('Fold sample universes differ.')
    if any(sum(f[c] == 'validation' for f in folds) != 1 for c in universe):
        raise ValueError('Each sample must be held out exactly once.')
    return paths, folds


def make_roles(folds, outer_fold):
    outer = folds[outer_fold - 1]
    return {code: ('outer_eval' if outer[code] == 'validation' else 'train')
            for code in sorted(outer)}


class MergeSingletonBatchSampler:
    """Keep all chunks; append a final singleton to the preceding batch."""
    def __init__(self, sampler, batch_size):
        if batch_size < 2 or len(sampler) < 2:
            raise ValueError('BT requires batch_size >= 2 and at least two chunks.')
        self.sampler, self.batch_size = sampler, batch_size

    def __len__(self):
        n, b = len(self.sampler), self.batch_size
        q, r = divmod(n, b)
        return q + int(r > 1)

    def __iter__(self):
        batch = []
        remaining = len(self.sampler)
        for index in self.sampler:
            batch.append(index)
            remaining -= 1
            if len(batch) >= self.batch_size and remaining != 1:
                yield batch
                batch = []
        if batch:
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split_dir', type=Path, default=Path(__file__).parent / 'k_cross')
    parser.add_argument('--outer_fold', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--output_root', type=Path, default=Path('runs_bt_fixed'))
    parser.add_argument('--dry_run', action='store_true', help='Check splits without importing PyTorch or writing files.')
    parser.add_argument('--samples_dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sequence_length', type=int, default=300)
    parser.add_argument('--sample_subset_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--token_emb_dim', type=int, default=8)
    parser.add_argument('--sample_repr_dim', type=int, default=256)
    parser.add_argument('--sample_emb_dim', type=int, default=64)
    parser.add_argument('--barlow_twins_lambda', type=float, default=0.005)
    parser.add_argument('--initial_learning_rate', type=float, default=0.001)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    args = parser.parse_args()
    paths, folds = load_folds(args.split_dir)
    roles = make_roles(folds, args.outer_fold)
    print(f'Outer fold {args.outer_fold}: {dict(Counter(roles.values()))}')
    if args.dry_run:
        return
    if args.samples_dir is None:
        parser.error('--samples_dir is required for training.')
    for key in ['sequence_length', 'sample_subset_size', 'batch_size', 'token_emb_dim',
                'sample_repr_dim', 'sample_emb_dim', 'initial_learning_rate', 'max_epochs']:
        if getattr(args, key) <= 0:
            parser.error(f'--{key} must be positive.')
    if args.sample_repr_dim % 4 or args.batch_size < 2 or args.num_workers < 0 or args.barlow_twins_lambda < 0 or args.weight_decay < 0:
        parser.error('repr_dim must be divisible by 4; batch_size >= 2; workers, lambda and weight_decay >= 0.')

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, RandomSampler
    from pytorch_lightning.loggers import CSVLogger
    from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset
    from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder

    pl.seed_everything(args.seed, workers=True)
    datasets = {}
    for role in ['train']:
        files = [args.samples_dir / f'{c}.csv' for c, r in roles.items() if r == role]
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {role} files: {missing}')
        dataset = BarlowTwinsDataset(files, args.sample_subset_size, args.sequence_length)
        excluded = set(files) - set(dataset.files)
        if excluded:
            raise ValueError(f'Samples excluded by Dataset (columns/read count): {sorted(map(str, excluded))}')
        datasets[role] = dataset
    loaders = {}
    for role, dataset in datasets.items():
        sampler = RandomSampler(dataset)
        loaders[role] = DataLoader(dataset, batch_sampler=MergeSingletonBatchSampler(sampler, args.batch_size),
                                   num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    run_dir = args.output_root / f'fold_{args.outer_fold:02d}' / f'seed_{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['split_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    config['sample_counts'] = dict(Counter(roles.values()))
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    with (run_dir / 'sample_roles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['spygen_code', 'role'])
        writer.writerows(roles.items())
    for path in paths:
        (run_dir / path.name).write_bytes(path.read_bytes())
    model = SelfAttentionBarlowTwinsEmbedder(
        token_emb_dim=args.token_emb_dim, seq_len=args.sequence_length,
        sample_repr_dim=args.sample_repr_dim, sample_emb_dim=args.sample_emb_dim,
        lmbda=args.barlow_twins_lambda, initial_learning_rate=args.initial_learning_rate,
        weight_decay=args.weight_decay)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, max_epochs=args.max_epochs,
                         logger=CSVLogger(str(run_dir), name='metrics'), enable_checkpointing=False,
                         limit_val_batches=0, num_sanity_val_steps=0,
                         log_every_n_steps=10, default_root_dir=str(run_dir))
    trainer.fit(model, train_dataloaders=loaders['train'])
    checkpoint_path = run_dir / 'final.ckpt'
    trainer.save_checkpoint(str(checkpoint_path))
    print(f'Final checkpoint (no validation selection): {checkpoint_path}')



if __name__ == '__main__':
    main()
