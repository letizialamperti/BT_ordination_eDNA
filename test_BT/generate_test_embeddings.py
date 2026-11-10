"""Generate test embeddings for the four frozen BT folds; no classification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DEFAULT_BT_REPO='/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bt-repo',type=Path,default=os.environ.get('BT_REPO',DEFAULT_BT_REPO))
    parser.add_argument('--runs-root',type=Path,default=os.environ.get('RUNS_ROOT'))
    parser.add_argument('--samples-dir',type=Path,default=os.environ.get('SAMPLES_DIR','/bettik/PROJECTS/pr-qiepb/lampertl/all'))
    parser.add_argument('--output-dir',type=Path,default=os.environ.get('OUTPUT_DIR',str(ROOT/'test_embeddings')))
    parser.add_argument('--batch-size',type=int,default=int(os.environ.get('EMBEDDING_BATCH_SIZE','64')))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    runs=args.runs_root or args.bt_repo/'runs_bt_432samples/cap_0'
    env=dict(os.environ,WANDB_MODE='disabled')
    env.setdefault('OMP_NUM_THREADS','1')
    env.setdefault('OPENBLAS_NUM_THREADS','1')
    common=['--bt-repo',str(args.bt_repo),'--runs-root',str(runs),'--samples-dir',str(args.samples_dir)]
    subprocess.run([sys.executable,str(ROOT/'preflight.py'),*common],env=env,check=True)
    if args.check_only:return
    for fold in [1,2,3,4]:
        print(f'Generating test embeddings: fold {fold:02d}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'extract_test_embeddings.py'),'--fold',str(fold),*common,
                        '--output-dir',str(args.output_dir),'--batch-size',str(args.batch_size)],env=env,check=True)
    print(f'Test embeddings saved in: {args.output_dir.resolve()}',flush=True)
    print('Copy the entire test_embeddings directory to the local computer for classification.',flush=True)

if __name__=='__main__':
    main()
