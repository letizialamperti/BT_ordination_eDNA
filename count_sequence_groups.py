#!/usr/bin/env python3
"""Stream paired-read CSVs once and report non-overlapping BT groups for each N.

Matches the missing-value policy of the corrected BarlowTwinsDataset.
No torch, pandas or GPU required. Input: one CSV record per physical line.
A row is a Forward/Reverse pair; one BT example requires 2*N valid rows.
Malformed files or unsupported/overlong reads are reported as errors, not
silently treated as usable. No input files are modified.
"""
import argparse
import csv
from pathlib import Path

MISSING = {'', 'na', 'nan', 'none', 'null', '<na>', 'n/a'}
ALLOWED = set('acgtryswkmbdhvnu')


def scan(path, sequence_length):
    counts = dict(total_rows=0, valid_paired_reads=0, discarded_missing_pairs=0)
    before = path.stat()
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            raw = stream.readline()
            if not raw:
                raise ValueError('empty_file')
            header = next(csv.reader([raw], strict=True))
            if header.count('Forward') != 1 or header.count('Reverse') != 1:
                raise ValueError('missing_or_duplicate_required_columns')
            fi, ri = header.index('Forward'), header.index('Reverse')
            for line, raw in enumerate(stream, 2):
                counts['total_rows'] += 1
                row = next(csv.reader([raw], strict=True))
                if not row:
                    row = [''] * len(header)
                if len(row) != len(header):
                    raise ValueError(f'line {line}: incorrect column count')
                pair = (row[fi].strip(), row[ri].strip())
                if any(read.lower() in MISSING for read in pair):
                    counts['discarded_missing_pairs'] += 1
                    continue
                if any(len(read) > sequence_length or set(read.lower()) - ALLOWED for read in pair):
                    raise ValueError(f'line {line}: unsupported nucleotide or read longer than {sequence_length}')
                counts['valid_paired_reads'] += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('input_changed_during_scan')
    except (ValueError, csv.Error, UnicodeError, OSError) as exc:
        # Partial counts must never be mistaken for a complete sample count.
        return {key: '' for key in counts}, str(exc)
    return counts, ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples_dir', type=Path, required=True)
    parser.add_argument('--subset_sizes', type=int, nargs='+', default=[250, 500, 1000])
    parser.add_argument('--sequence_length', type=int, default=300)
    parser.add_argument('--output_dir', type=Path, default=Path('sequence_group_counts'))
    args = parser.parse_args()
    sizes = sorted(set(args.subset_sizes))
    if min(sizes) <= 0 or args.sequence_length <= 0:
        parser.error('Subset sizes and sequence length must be positive.')
    paths = sorted(args.samples_dir.glob('*.csv'))
    if not paths:
        parser.error(f'No CSV files found in {args.samples_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    totals = {n: dict(subset_size=n, files=len(paths), usable_samples=0,
                     insufficient_samples=0, error_files=0, bt_groups=0,
                     views_used=0, used_valid_pairs=0, unused_valid_pairs=0) for n in sizes}
    fields = ['sample', 'file', 'subset_size', 'total_rows', 'valid_paired_reads',
              'discarded_missing_pairs', 'pairs_per_bt_group', 'bt_groups',
              'views_used', 'used_valid_pairs', 'unused_valid_pairs', 'status', 'error']
    with (args.output_dir / 'groups_per_sample.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(paths, 1):
            counts, error = scan(path, args.sequence_length)
            for n in sizes:
                row = dict(sample=path.stem, file=str(path.resolve()), subset_size=n,
                           **counts, pairs_per_bt_group=2*n, error=error)
                summary = totals[n]
                if error:
                    row['status'] = 'error'
                    summary['error_files'] += 1
                else:
                    groups, remainder = divmod(counts['valid_paired_reads'], 2*n)
                    row.update(bt_groups=groups, views_used=2*groups,
                               used_valid_pairs=2*n*groups, unused_valid_pairs=remainder,
                               status='usable' if groups else 'insufficient')
                    summary['usable_samples' if groups else 'insufficient_samples'] += 1
                    for key in ['bt_groups', 'views_used', 'used_valid_pairs', 'unused_valid_pairs']:
                        summary[key] += row[key]
                writer.writerow(row)
            print(f'[{index}/{len(paths)}] {path.name}: {error or str(counts["valid_paired_reads"]) + " valid pairs"}', flush=True)
    with (args.output_dir / 'summary.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(totals[sizes[0]]))
        writer.writeheader()
        writer.writerows(totals.values())
    print(f'Reports: {args.output_dir.resolve()}')
    if any(total['error_files'] for total in totals.values()):
        print('ERROR: some files could not be counted; see error column. Summary excludes these files.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
