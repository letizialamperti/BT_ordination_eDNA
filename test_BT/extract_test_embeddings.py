"""Apply frozen BT to28test samples; reproduce a training anchor to verify encoder/classifier pairing."""
import argparse,importlib.util,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from inference_common import ROOT,FEATURES,load_bundle,check_run,sha256,read_json,save_json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',type=int,choices=range(1,5),required=True);p.add_argument('--bt-repo',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--samples-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
assert a.batch_size>0
m,samples,_=load_bundle(extraction_only=True);info=m['folds'][str(a.fold)];run=a.runs_root/f'fold_{a.fold:02d}/seed_0';check_run(run,a.fold,m,samples)
out=a.output_dir/f'fold_{a.fold:02d}';out.mkdir(parents=True,exist_ok=True)
checkpoint=run/'final.ckpt';ids=samples.spygen_code.tolist();anchor=info['anchor_id']
inputs={code:{'size':(a.samples_dir/f'{code}.csv').stat().st_size,'mtime_ns':(a.samples_dir/f'{code}.csv').stat().st_mtime_ns} for code in [anchor]+ids}
provenance={'fold':a.fold,'bundle_sha256':sha256(ROOT/'manifest.json'),'checkpoint_path':str(checkpoint.resolve()),'checkpoint_sha256':sha256(checkpoint),'input_file_stats':inputs,'extraction_batch_size':a.batch_size,'protocol':m['inference_protocol']}
for relative in ['ORDNA/models/barlow_twins.py','ORDNA/models/representation_module.py','ORDNA/utils/sequence_mapper.py','ORDNA/utils/constants.py']:
 provenance.setdefault('BT_source_files_sha256',{})[relative]=sha256(a.bt_repo/relative)
provenance_path=out/'extraction_manifest.json'
if provenance_path.exists():
 previous=read_json(provenance_path)
 if all(previous.get(k)==v for k,v in provenance.items()) and previous.get('anchor_verified') and all(sha256(out/k)==v for k,v in previous['output_sha256'].items()):
  print(f'Fold{a.fold}: verified cached embeddings, skipping extraction.');sys.exit(0)
 raise ValueError('Existing extraction has different inputs/configuration; choose a new OUTPUT_DIR.')
# The vendored generator is the exact protocol used for reference embeddings.
# BT architecture and mapper are imported from the user-selected BT repository.
sys.path.insert(0,str(a.bt_repo.resolve()))
spec=importlib.util.spec_from_file_location('bt_frozen_extraction',ROOT/'vendor_bt/generate_embeddings_fold.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='.extract_',dir=out) as temporary:
 work=Path(temporary);roles=pd.concat([pd.DataFrame({'spygen_code':[anchor],'role':['reference_check']}),samples],ignore_index=True)
 roles.to_csv(work/'sample_roles.csv',index=False)
 module.RUN_DIR=run;module.CHECKPOINT_PATH=checkpoint;module.SAMPLE_ROLES_PATH=work/'sample_roles.csv';module.SAMPLE_DIR=a.samples_dir
 module.SEQUENCE_LENGTH=300;module.INFERENCE_SUBSET_SIZE=500;module.EMBEDDING_BATCH_SIZE=a.batch_size;module.RANDOM_SEED=0;module.OUTPUT_DIR=work
 module.PREPROJECTOR_OUTPUT=work/'embeddings_preprojector_256D.csv';module.PROJECTOR_OUTPUT=work/'embeddings_projector_64D.csv';module.QC_OUTPUT=work/'embedding_qc.csv'
 module.main()
 pre=pd.read_csv(module.PREPROJECTOR_OUTPUT).set_index('Sample');qc=pd.read_csv(module.QC_OUTPUT).set_index('Sample')
 if set(pre.index)!=set(ids+[anchor]):raise ValueError('Some test samples lack a complete500-pair subset; no partial test set accepted.')
 actual=pre.loc[anchor,FEATURES].to_numpy(dtype='float32');expected=np.asarray(info['anchor_mean'],dtype='float32')
 for key,value in info['anchor_qc'].items():
  if int(qc.loc[anchor,key])!=value:raise ValueError(f'Training anchor QC changed: {anchor}/{key}')
 difference=float(np.max(np.abs(actual-expected)))
 if not np.allclose(actual,expected,rtol=1e-4,atol=1e-4):
  save_json(out/'anchor_mismatch.json',{'anchor':anchor,'max_absolute_error':difference,'expected':expected.tolist(),'actual':actual.tolist()})
  raise ValueError('BT checkpoint does not reproduce its frozen training anchor. Do not pair this encoder with these classifiers.')
 for filename in ['embeddings_preprojector_256D.csv','embeddings_projector_64D.csv','embedding_qc.csv']:
  frame=pd.read_csv(work/filename).set_index('Sample').loc[ids].reset_index()
  assert frame.role.eq('test').all() and len(frame)==28
  frame.to_csv(out/filename,index=False)
 provenance.update(anchor_verified=True,anchor_id=anchor,anchor_max_absolute_error=difference)
 provenance['output_sha256']={name:sha256(out/name) for name in ['embeddings_preprojector_256D.csv','embeddings_projector_64D.csv','embedding_qc.csv']}
 save_json(provenance_path,provenance)
print(f'Fold{a.fold}:28test embeddings, anchor verified (max difference{difference:.3g}).')
