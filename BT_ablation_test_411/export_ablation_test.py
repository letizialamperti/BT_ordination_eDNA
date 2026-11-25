"""Export shared untrained embeddings or fold-matched trained test embeddings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import torch
import export_final as api
import export_io as io
from export_staged import process
from final_checks import sha, file_stats, read_roles, run_lock

HERE=Path(__file__).resolve().parent


def state_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_run(runs,fold):
    run=runs/f"fold_{fold:02d}/seed_0"
    cfg=io.read_json(run/'config.json');recipe=json.loads((HERE/'frozen_configuration.json').read_text())
    for key,value in recipe['encoder'].items():
        if cfg.get(key)!=value:raise ValueError(f"Fold {fold}: recipe differs in {key}")
    role_path=HERE/f'splits/fold_{fold:02d}.csv';roles=read_roles(role_path,fold)
    if cfg['roles_sha256']!=sha(role_path) or cfg.get('smoke_test'):
        raise ValueError(f"Fold {fold}: incorrect cohort or smoke-test run")
    expected=set(roles.loc[roles.role.eq('train'),'spygen_code'])
    if {Path(p).stem for p in cfg['raw_input_stats']}!=expected:
        raise ValueError(f"Fold {fold}: pretraining cohort mismatch")
    return run,cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['initial','trained-test'],required=True)
    p.add_argument('--fold',type=int,choices=[1,2,3,4])
    p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--samples-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--scratch-dir',type=Path,default=os.environ.get('BT_SHUFFLE_SCRATCH'))
    args=p.parse_args()
    args.batch_size=64
    if args.kind=='trained-test' and args.fold is None:p.error('--fold is required for trained-test')
    cohort=pd.read_csv(HERE/'cohort_411.csv');tests=pd.read_csv(HERE/'test_samples.csv')
    if len(cohort)!=411 or not cohort.spygen_code.is_unique or len(tests)!=28 or not tests.spygen_code.is_unique:
        raise ValueError('Expected 411 cohort IDs and 28 unique test IDs')
    if set(cohort.spygen_code)&set(tests.spygen_code):raise ValueError('Test and cohort overlap')
    if set(tests.role)!={'test'}:raise ValueError('Invalid test roles')
    name='initial_shared' if args.kind=='initial' else f'trained_fold{args.fold:02d}'
    context=args.output_root/name
    with run_lock(context):
        stage=Path(tempfile.mkdtemp(prefix='bt_ablation_',dir=args.scratch_dir));success=False
        try:
            identities=[]
            folds=[1,2,3,4] if args.kind=='initial' else [args.fold]
            checkpoint=None;config=None
            for fold in folds:
                run,cfg=inspect_run(args.runs_root,fold)
                source=run/('initial.ckpt' if args.kind=='initial' else 'final.ckpt')
                digest=io.retry(lambda:sha(source),f'hash {source}')
                if args.kind=='initial':
                    local=stage/f'initial_{fold}.ckpt';io.copy_verified(source,local,expected_sha=digest)
                    ckpt=torch.load(local,map_location='cpu',weights_only=False)
                    if ckpt.get('global_step')!=0:raise ValueError('Initial checkpoint has training steps')
                    identities.append(dict(fold=fold,checkpoint_sha256=digest,state_dict_sha256=state_hash(ckpt['state_dict'])))
                    del ckpt
                    if fold!=1:local.unlink()
                else:
                    done=io.read_json(run/'training_complete.json')
                    if done['completed_epochs']!=6 or done['checkpoint_sha256']!=digest or done['config_sha256']!=io.retry(lambda:sha(run/'config.json'),'hash config'):
                        raise ValueError('Final six-epoch checkpoint not verified')
                    identities.append(dict(fold=fold,checkpoint_sha256=digest))
                if checkpoint is None:checkpoint=source;config=cfg
            if args.kind=='initial' and len({r['state_dict_sha256'] for r in identities})!=1:
                raise ValueError('Initial weights/buffers differ: shared export forbidden; export each initialization separately')
            roles=tests[['spygen_code','role']].copy()
            if args.kind=='initial':
                c=cohort[['spygen_code']].copy();c['role']='cohort';roles=pd.concat([c,roles],ignore_index=True)
            roles=roles.sort_values('spygen_code').reset_index(drop=True)
            paths=[args.samples_dir/f'{code}.csv' for code in roles.spygen_code]
            stats=io.retry(lambda:file_stats(paths),'stat export input files')
            output=context/'embeddings';io.retry(lambda:output.mkdir(parents=True,exist_ok=True),'create output')
            expected=dict(kind=args.kind,initial_state_verified_shared=args.kind=='initial',checkpoint_identities=identities,
                          roles=roles.to_dict('records'),source_stats=stats,batch_size=64,
                          exporter_sha256=sha(__file__),generator_sha256=sha(HERE/'vendor_bt/generate_embeddings_fold.py'),
                          io_sources={n:sha(HERE/n) for n in ['export_final.py','export_staged.py','export_io.py']})
            protocol=output/'export_protocol.json'
            if io.retry(protocol.exists,'check export protocol') and io.read_json(protocol)!=expected:
                raise ValueError('Existing export differs; preserve it and use a separate output root')
            io.write_json(protocol,expected)
            marker=output/'manifest.json'
            if io.retry(marker.exists,'check complete export'):
                api.verified(output,io.read_json(marker),expected)
                print(f'Verified complete export: {output}',flush=True)
            else:
                process(args,api,context,roles,config,stats,paths,checkpoint,identities[0]['checkpoint_sha256'],output,expected,stage)
            success=True
        finally:
            if success:shutil.rmtree(stage)
            else:print(f'Local diagnostic staging retained: {stage}',flush=True)


if __name__=='__main__':main()
