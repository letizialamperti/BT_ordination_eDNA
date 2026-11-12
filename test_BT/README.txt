Pacchetto server: estrazione embedding BT432 dei 28 campioni test.

Caricare tutta questa cartella in:
/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/test_BT

Checkpoint predefiniti (quattro fold, seed 0):
/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/runs_bt_432samples/cap_0/fold_XX/seed_0/final.ckpt
Ogni run deve contenere anche config.json e sample_roles.csv.
Gli embedding di sviluppo corrispondenti si trovano in seed_0/embeddings.

Dalla cartella test_BT sul server:
  bash run_test.sh --check-only
  bash run_test.sh
Oppure:
  oarsub -S ./run_test.sh

Output predefinito: test_BT/test_embeddings_BT432/fold_01 ... fold_04.
Ogni fold contiene embedding 256D e 64D, QC ed extraction_manifest.json.
Scaricare l'intera cartella di output per la classificazione locale.

Il preflight verifica i 432 ruoli e l'assenza dei 28 test nel training.
I riferimenti nel manifest provengono dalla cartella locale embeddings432:
432 campioni nel QC, 430 embedding validi (due campioni hanno meno di 500 coppie).
Per ogni fold viene riprodotto un campione di training e confrontato con il
suo embedding di riferimento, prima di salvare gli embedding test.
Il protocollo usa sequenze di lunghezza 300, subset completi di 500 coppie,
shuffle seed 0 e media su tutti i subset completi; scarta il resto.

Questo pacchetto esegue solo estrazione, senza classificazione.
I classificatori da applicare devono essere addestrati sugli embedding BT432
corrispondenti a questi checkpoint.
Reads predefiniti: /bettik/PROJECTS/pr-qiepb/lampertl/all.
Ambiente predefinito: zioboia.

Per cambiare i percorsi senza modificare i file verificati dal manifest:
  bash run_test.sh --runs-root /percorso/runs_bt_432samples/cap_0
  bash run_test.sh --output-dir /percorso/test_embeddings_BT432
Sono supportate anche BT_REPO, RUNS_ROOT, SAMPLES_DIR e OUTPUT_DIR.
Usare una nuova cartella di output se contiene estrazioni dei vecchi checkpoint.
