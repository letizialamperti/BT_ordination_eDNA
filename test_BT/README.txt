Pacchetto server: sola estrazione embedding BT dei28campioni test.

Caricare questa cartella in:
/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/test_BT

Dalla cartella test_BT sul server:
  bash run_test.sh --check-only
  bash run_test.sh
Oppure:
  oarsub -S ./run_test.sh

Output predefinito: test_BT/test_embeddings/fold_01 ... fold_04.
Scaricare l'intera cartella test_embeddings, inclusi QC ed extraction_manifest.json,
nella cartella locale Classifier_BT_SL_WP3_DNAi/test/ per la classificazione.

Questo pacchetto non contiene classificatori o etichette e non esegue classificazione.
Usa i checkpoint del BT411 in runs_bt_bs64/cap_0/fold_XX/seed_0/final.ckpt.
Reads predefiniti: /bettik/PROJECTS/pr-qiepb/lampertl/all.
Ambiente predefinito: zioboia.
Non modificare i file protetti dal manifest; usare le variabili d'ambiente per i percorsi.

run_test.sh attiva zioboia ed esegue solamente:
  python generate_test_embeddings.py "$@"

Il file Python gestisce preflight e ciclo sui quattro fold. Per cambiare i percorsi:
  bash run_test.sh --output-dir /percorso/test_embeddings

