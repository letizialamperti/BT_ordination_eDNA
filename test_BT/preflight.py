import argparse,sys
from pathlib import Path
from inference_common import load_bundle,check_run
p=argparse.ArgumentParser();p.add_argument('--bt-repo',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--samples-dir',type=Path,required=True);a=p.parse_args()
m,samples,_=load_bundle(extraction_only=True)
for file in ['ORDNA/models/barlow_twins.py','ORDNA/utils/sequence_mapper.py']:
 if not (a.bt_repo/file).is_file():raise FileNotFoundError(a.bt_repo/file)
sys.path.insert(0,str(a.bt_repo.resolve()))
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper
ids=set(samples.spygen_code)
for fold in range(1,5):
 run=a.runs_root/f'fold_{fold:02d}/seed_0';cfg=check_run(run,fold,m,samples)
 ids.add(m['folds'][str(fold)]['anchor_id'])
 print(f'OK fold{fold}: final.ckpt,411roles, configuration, no test overlap',flush=True)
for code in sorted(ids):
 if not (a.samples_dir/f'{code}.csv').is_file():raise FileNotFoundError(a.samples_dir/f'{code}.csv')
print('Embedding-only preflight passed. No classifiers or test labels are needed on this server. Checkpoint pairing will be checked with a frozen training anchor.')
