import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper


# ============================================================
# Configuration
# ============================================================

RUN_DIR = Path("runs_bt_bs64/cap_0/fold_04/seed_0")
CHECKPOINT_PATH = RUN_DIR / "final.ckpt"
SAMPLE_ROLES_PATH = RUN_DIR / "sample_roles.csv"

SAMPLE_DIR = Path("/bettik/PROJECTS/pr-qiepb/lampertl/all")

SEQUENCE_LENGTH = 300

# The BT model was trained with views containing 500 paired reads.
INFERENCE_SUBSET_SIZE = 500

# Number of 500-read subsets processed simultaneously on the GPU.
# BS64 was stable during training, so 64 is a sensible starting point.
EMBEDDING_BATCH_SIZE = 64

RANDOM_SEED = 0

OUTPUT_DIR = RUN_DIR / "embeddings"

PREPROJECTOR_OUTPUT = OUTPUT_DIR / "embeddings_preprojector_256D.csv"
PROJECTOR_OUTPUT = OUTPUT_DIR / "embeddings_projector_64D.csv"
QC_OUTPUT = OUTPUT_DIR / "embedding_qc.csv"


# ============================================================
# Helpers
# ============================================================

def clean_sample_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Read Forward/Reverse paired reads and apply the same basic QC
    used for BT input preparation.
    """
    df = pd.read_csv(
        file_path,
        usecols=["Forward", "Reverse"],
    )

    original_rows = len(df)

    # Remove NaN pairs.
    df = df.dropna(subset=["Forward", "Reverse"])

    # Remove empty/whitespace-only sequences.
    valid_mask = (
        df["Forward"].astype(str).str.strip().ne("")
        & df["Reverse"].astype(str).str.strip().ne("")
    )

    df = df.loc[valid_mask].reset_index(drop=True)

    return df, original_rows


def build_input_tensor(
    subset_df: pd.DataFrame,
    sequence_mapper: SequenceMapper,
) -> torch.Tensor:
    """
    Convert one subset of paired reads into a tensor with shape:

        [subset_size, 2, sequence_length]
    """

    forward_sequences = subset_df["Forward"].tolist()
    reverse_sequences = subset_df["Reverse"].tolist()

    forward_tensor = torch.tensor(
        sequence_mapper.map_seq_list(
            forward_sequences,
            pad_to_len=SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
    )

    reverse_tensor = torch.tensor(
        sequence_mapper.map_seq_list(
            reverse_sequences,
            pad_to_len=SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
    )

    return torch.stack(
        (forward_tensor, reverse_tensor),
        dim=1,
    )


def write_embedding_csv(
    output_path: Path,
    records,
    embedding_dim: int,
):
    """
    Save mean and standard deviation of subset embeddings.
    """

    header = [
        "Sample",
        "role",
    ]

    header += [
        f"Dim{i + 1}"
        for i in range(embedding_dim)
    ]

    header += [
        f"Standard_dev_{i + 1}"
        for i in range(embedding_dim)
    ]

    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)

        for record in records:
            row = [
                record["Sample"],
                record["role"],
            ]

            row += record["mean"].tolist()
            row += record["std"].tolist()

            writer.writerow(row)


# ============================================================
# Main
# ============================================================

def main():

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not SAMPLE_ROLES_PATH.is_file():
        raise FileNotFoundError(
            f"sample_roles.csv not found: {SAMPLE_ROLES_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}", flush=True)

    # --------------------------------------------------------
    # Load frozen BT model
    # --------------------------------------------------------

    model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(
        CHECKPOINT_PATH
    )

    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    print(
        f"Loaded checkpoint: {CHECKPOINT_PATH}",
        flush=True,
    )

    # --------------------------------------------------------
    # Read exactly the samples belonging to this outer fold
    # --------------------------------------------------------

    roles_df = pd.read_csv(SAMPLE_ROLES_PATH)

    required_columns = {"spygen_code", "role"}

    if not required_columns.issubset(roles_df.columns):
        raise ValueError(
            f"{SAMPLE_ROLES_PATH} must contain "
            f"{sorted(required_columns)}"
        )

    print(
        f"Samples to embed: {len(roles_df)}",
        flush=True,
    )

    print(
        roles_df["role"].value_counts().to_dict(),
        flush=True,
    )

    # --------------------------------------------------------
    # Mapper
    # --------------------------------------------------------

    sequence_mapper = SequenceMapper()

    preprojector_records = []
    projector_records = []
    qc_records = []

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    for sample_idx, row in roles_df.iterrows():

        sample_code = str(row["spygen_code"])
        role = str(row["role"])

        file_path = SAMPLE_DIR / f"{sample_code}.csv"

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing sample file: {file_path}"
            )

        sample_df, original_rows = clean_sample_dataframe(
            file_path
        )

        n_valid_pairs = len(sample_df)
        n_discarded = original_rows - n_valid_pairs

        # ----------------------------------------------------
        # Deterministic shuffle before partitioning into subsets
        # ----------------------------------------------------

        sample_df = sample_df.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        # Only complete subsets.
        n_full_chunks = (
            n_valid_pairs // INFERENCE_SUBSET_SIZE
        )

        n_used_pairs = (
            n_full_chunks * INFERENCE_SUBSET_SIZE
        )

        n_leftover_pairs = (
            n_valid_pairs - n_used_pairs
        )

        print(
            f"[{sample_idx + 1}/{len(roles_df)}] "
            f"{sample_code} ({role}): "
            f"{n_valid_pairs} valid pairs, "
            f"{n_full_chunks} full subsets, "
            f"{n_leftover_pairs} leftover",
            flush=True,
        )

        # ----------------------------------------------------
        # Samples too small to produce a complete inference view
        # ----------------------------------------------------

        if n_full_chunks == 0:

            print(
                f"WARNING: {sample_code} has fewer than "
                f"{INFERENCE_SUBSET_SIZE} valid pairs; "
                f"no embedding produced.",
                flush=True,
            )

            qc_records.append(
                {
                    "Sample": sample_code,
                    "role": role,
                    "original_rows": original_rows,
                    "valid_pairs": n_valid_pairs,
                    "discarded_pairs": n_discarded,
                    "full_subsets": 0,
                    "used_pairs": 0,
                    "leftover_pairs": n_valid_pairs,
                    "embedding_status": "excluded_no_full_subset",
                }
            )

            continue

        # ----------------------------------------------------
        # Embedding accumulators
        # ----------------------------------------------------

        preprojector_embeddings = []
        projector_embeddings = []

        # ----------------------------------------------------
        # Process several subsets simultaneously
        # ----------------------------------------------------

        for chunk_batch_start in range(
            0,
            n_full_chunks,
            EMBEDDING_BATCH_SIZE,
        ):

            chunk_batch_end = min(
                chunk_batch_start + EMBEDDING_BATCH_SIZE,
                n_full_chunks,
            )

            input_chunks = []

            for chunk_idx in range(
                chunk_batch_start,
                chunk_batch_end,
            ):

                start = (
                    chunk_idx * INFERENCE_SUBSET_SIZE
                )

                end = (
                    start + INFERENCE_SUBSET_SIZE
                )

                subset_df = sample_df.iloc[start:end]

                input_chunks.append(
                    build_input_tensor(
                        subset_df,
                        sequence_mapper,
                    )
                )

            # Shape:
            # [B_chunks, 500, 2, sequence_length]
            input_tensor = torch.stack(
                input_chunks,
                dim=0,
            ).to(device, non_blocking=True)

            # ------------------------------------------------
            # Frozen inference
            # ------------------------------------------------

            with torch.inference_mode():

                # Representation before BT projector:
                # [B_chunks, 256]
                sample_repr = model.repr_module(
                    input_tensor
                )

                # Representation after BT projector:
                # [B_chunks, 64]
                sample_emb = model.out_mlp(
                    sample_repr
                )

            preprojector_embeddings.append(
                sample_repr.cpu().numpy()
            )

            projector_embeddings.append(
                sample_emb.cpu().numpy()
            )

        # ----------------------------------------------------
        # Concatenate all subsets for this biological sample
        # ----------------------------------------------------

        preprojector_embeddings = np.concatenate(
            preprojector_embeddings,
            axis=0,
        )

        projector_embeddings = np.concatenate(
            projector_embeddings,
            axis=0,
        )

        # ----------------------------------------------------
        # Mean and within-sample SD across subsets
        # ----------------------------------------------------

        pre_mean = np.mean(
            preprojector_embeddings,
            axis=0,
        )

        pre_std = np.std(
            preprojector_embeddings,
            axis=0,
        )

        proj_mean = np.mean(
            projector_embeddings,
            axis=0,
        )

        proj_std = np.std(
            projector_embeddings,
            axis=0,
        )

        preprojector_records.append(
            {
                "Sample": sample_code,
                "role": role,
                "mean": pre_mean,
                "std": pre_std,
            }
        )

        projector_records.append(
            {
                "Sample": sample_code,
                "role": role,
                "mean": proj_mean,
                "std": proj_std,
            }
        )

        qc_records.append(
            {
                "Sample": sample_code,
                "role": role,
                "original_rows": original_rows,
                "valid_pairs": n_valid_pairs,
                "discarded_pairs": n_discarded,
                "full_subsets": n_full_chunks,
                "used_pairs": n_used_pairs,
                "leftover_pairs": n_leftover_pairs,
                "embedding_status": "ok",
            }
        )

    # --------------------------------------------------------
    # Save both representations
    # --------------------------------------------------------

    write_embedding_csv(
        PREPROJECTOR_OUTPUT,
        preprojector_records,
        embedding_dim=256,
    )

    write_embedding_csv(
        PROJECTOR_OUTPUT,
        projector_records,
        embedding_dim=64,
    )

    pd.DataFrame(qc_records).to_csv(
        QC_OUTPUT,
        index=False,
    )

    print()
    print("Embedding extraction complete.", flush=True)

    print(
        f"Pre-projector 256D: {PREPROJECTOR_OUTPUT}",
        flush=True,
    )

    print(
        f"Projector 64D:     {PROJECTOR_OUTPUT}",
        flush=True,
    )

    print(
        f"QC report:         {QC_OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
