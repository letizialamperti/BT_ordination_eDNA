"""Synthetic export fault injection; never reads real study data."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import export_final as api
import export_io as io
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from final_checks import sha, file_stats


def main():
    checks=[];torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="bt_export_io_test_") as temporary:
        root=Path(temporary);run=root/"run";run.mkdir();samples=root/"reads";samples.mkdir()
        records=[dict(spygen_code=f"sample_{block}",spatial_block=block,
                      role="outer_eval" if block==1 else "train") for block in [1,2,3,4]]
        roles=pd.DataFrame(records);roles.to_csv(run/"sample_roles.csv",index=False)
        for row in records:
            pairs=[]
            for i in range(17):
                seq=''.join('ACGT'[(i//(4**j))%4] for j in [3,2,1,0])
                pairs.append(dict(Forward=seq,Reverse=seq[::-1]))
            pd.DataFrame(pairs).to_csv(samples/f"{row['spygen_code']}.csv",index=False)
        cfg=dict(outer_fold=1,smoke_test=True,views="fixed",max_epochs=2,sequence_length=8,sample_subset_size=2,
                 roles_sha256=sha(run/"sample_roles.csv"),
                 raw_input_stats=file_stats([samples/f"sample_{b}.csv" for b in [2,3,4]]))
        (run/"config.json").write_text(json.dumps(cfg))
        torch.manual_seed(0)
        model=SelfAttentionBarlowTwinsEmbedder(token_emb_dim=2,seq_len=8,sample_repr_dim=256,
              sample_emb_dim=64,lmbda=.005,initial_learning_rate=.0003,weight_decay=.0001)
        torch.save(dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),
                        **{'pytorch-lightning_version':pl.__version__},epoch=1,global_step=1),run/"final.ckpt")
        (run/"training_complete.json").write_text(json.dumps(dict(completed_epochs=2,
            checkpoint_sha256=sha(run/"final.ckpt"),config_sha256=sha(run/"config.json"))))
        args=SimpleNamespace(run_dir=run,samples_dir=samples,batch_size=3,scratch_dir=root)
        original=Path.open;failures=[0]
        def flaky(self,*a,**k):
            if self==samples/"sample_2.csv" and a and a[0]=='rb' and failures[0]<2:
                failures[0]+=1;raise OSError(70,'Communication error on send')
            return original(self,*a,**k)
        with patch.object(Path,'open',flaky),patch.object(io.time,'sleep'):
            api.export(args)
        assert failures[0]==2
        out=run/"embeddings/epoch_02";first=pd.read_csv(out/"embeddings_preprojector_256D.csv")
        assert len(first)==4
        checks.append('Real-model export recovers from injected Errno 70 while staging raw reads')
        bad=out/"_samples/sample_2/embeddings_preprojector_256D.csv"
        bad.write_text(bad.read_text()+'\n')
        (out/"manifest.json").unlink()
        with patch.object(io.time,'sleep'):
            api.export(args)
        repaired=pd.read_csv(out/"embeddings_preprojector_256D.csv")
        assert np.array_equal(first.filter(regex='^Dim').to_numpy(),repaired.filter(regex='^Dim').to_numpy())
        assert len(list((out/"_quarantine").glob('sample_2_*')))==1
        checks.append('Entire cache audited: persistent bad hash quarantined, only bad sample regenerated, values identical')
        (out/"manifest.json").unlink()
        cache_file=out/"_samples/sample_3/embeddings_projector_64D.csv"
        def unavailable(self,*a,**k):
            if self==cache_file and a and a[0]=='rb':raise OSError(70,'Communication error on send')
            return original(self,*a,**k)
        with patch.object(Path,'open',unavailable),patch.object(io.time,'sleep'):
            try:api.export(args)
            except OSError as exc:assert exc.errno==70
            else:raise AssertionError('Persistent communication failure was ignored')
        assert (out/'_samples/sample_3/manifest.json').is_file()
        assert not list((out/'_quarantine').glob('sample_3_*'))
        checks.append('Persistent transport failure stops safely and does not quarantine a valid unreachable cache')
    report=dict(status='PASS',checks=checks)
    (api.HERE/'io_integration_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
