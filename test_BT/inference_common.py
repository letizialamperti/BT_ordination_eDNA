"""Inference-only helpers: no optimizer, training loop, or fitted preprocessing."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
FEATURES=[f'Dim{i}' for i in range(1,257)]
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_bundle(root=ROOT,extraction_only=False):
 root=Path(root);m=read_json(root/'manifest.json')
 extraction_files={'test_samples.csv','vendor_bt/generate_embeddings_fold.py','inference_common.py','preflight.py','extract_test_embeddings.py','generate_test_embeddings.py','run_test.sh'}
 for name,expected in m['files'].items():
  if extraction_only and name not in extraction_files:continue
  p=(root/name).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha256(p)!=expected:raise ValueError(f'Frozen bundle missing or changed: {name}')
 samples=pd.read_csv(root/m['samples_csv'],dtype={'spygen_code':str})
 assert len(samples)==m['n_test_samples']==28 and samples.spygen_code.is_unique and set(samples.role)=={'test'}
 expected={(f,t,s) for f in range(1,5) for t in m['targets'] for s in [0,1,2]}
 assert extraction_only or len(m['models'])==24 and {(r['fold'],r['target'],r['seed']) for r in m['models']}==expected
 for f,info in m['folds'].items():
  assert len(info['expected_roles'])==432 and not set(samples.spygen_code)&set(info['expected_roles'])
 if extraction_only:return m,samples,None
 labels=pd.read_csv(root/m['labels_csv'],dtype={'spygen_code':str})
 assert len(labels)==28 and labels.spygen_code.is_unique and samples.spygen_code.tolist()==labels.spygen_code.tolist()
 for t,info in m['targets'].items():
  encoded=labels[t].map({v:i for i,v in enumerate(info['class_values'])})
  assert not encoded.isna().any() and encoded.tolist()==labels[info['encoded_column']].tolist()
 return m,samples,labels

def check_run(run_dir,fold,manifest,samples):
 run_dir=Path(run_dir)
 for name in ['final.ckpt','config.json','sample_roles.csv']:
  if not (run_dir/name).is_file():raise FileNotFoundError(run_dir/name)
 cfg=read_json(run_dir/'config.json');roles=pd.read_csv(run_dir/'sample_roles.csv',dtype={'spygen_code':str})
 assert roles.spygen_code.is_unique
 if roles.set_index('spygen_code').role.to_dict()!=manifest['folds'][str(fold)]['expected_roles']:raise ValueError(f'Fold{fold}: encoder roles do not match the BT432 fold. Use the matching spatial training runs.')
 assert not set(samples.spygen_code)&set(roles.spygen_code)
 for key,wanted in {'outer_fold':fold,'seed':0,'sequence_length':300,'sample_subset_size':500,'sample_repr_dim':256,'sample_emb_dim':64}.items():
  if cfg.get(key)!=wanted:raise ValueError(f'{run_dir}: expected {key}={wanted}, found {cfg.get(key)}')
 if cfg.get('max_chunks_per_sample',0)!=0:raise ValueError(f'{run_dir}: expected an uncapped BT432 run')
 return cfg

def predict_one(model_info,x,root=ROOT):
 import joblib
 folder=Path(root)/model_info['directory'];cfg=read_json(folder/'config.json')
 assert cfg['head']=='softmax' and cfg['scaling']=='l2' and cfg['normalization']=='none'
 assert cfg['input_dim']==256 and cfg['hidden']==32 and cfg['input_features']==FEATURES
 assert cfg['fold']==model_info['fold'] and cfg['seed']==model_info['seed'] and cfg['target_column']==model_info['target']
 scaler=joblib.load(folder/'input_transform.joblib')
 assert scaler.norm=='l2' and scaler.n_features_in_==256
 xx=scaler.transform(np.asarray(x,dtype='float32')).astype('float32')
 assert np.isfinite(xx).all()
 model=nn.Sequential(nn.Linear(256,32),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(32,cfg['num_classes']))
 model.load_state_dict(torch.load(folder/'best_state.pt',map_location='cpu',weights_only=True),strict=True)
 model.eval().requires_grad_(False)
 with torch.inference_mode():prob=model(torch.from_numpy(xx)).softmax(1).numpy()
 assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
 return prob

def score(y,prob,class_values):
 from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
 pred=prob.argmax(1);k=len(class_values)
 return {'n_samples':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
 'macro_f1':float(f1_score(y,pred,labels=list(range(k)),average='macro',zero_division=0)),
 'class_values':class_values,'recall_by_class':recall_score(y,pred,labels=list(range(k)),average=None,zero_division=0).tolist(),
 'precision_by_class':precision_score(y,pred,labels=list(range(k)),average=None,zero_division=0).tolist(),
 'confusion_matrix':confusion_matrix(y,pred,labels=list(range(k))).tolist()}

def write_predictions(path,ids,prob,class_values,labels=None):
 pred=prob.argmax(1)
 out=pd.DataFrame({'spygen_code':ids,'prediction_encoded':pred,'prediction':[class_values[i] for i in pred]})
 for c,v in enumerate(class_values):out[f'probability_class_{v}']=prob[:,c]
 if labels is not None:out['label_encoded']=labels;out['label']=[class_values[i] for i in labels]
 out.to_csv(path,index=False)
