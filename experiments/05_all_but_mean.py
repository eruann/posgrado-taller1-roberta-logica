#!/usr/bin/env python
"""
Applies 'All-but-mean' normalization to RoBERTa embeddings stored in Parquet format.
This script is designed to run entirely on the GPU using cuDF and cuPy.

Assumes the input Parquet file has a 'wide' format, where each feature is a separate
column (e.g., 'feature_0', 'feature_1', ..., 'feature_N') and there is a 'label' column.

Example:
    python experiments/05_all_but_mean.py \\
        --source_path data/snli/embeddings/embeddings_snli_layer_12.parquet \\
        --out_path data/snli/embeddings/embeddings_snli_layer_12_normalized.parquet \\
        --experiment_name all-but-mean-normalization \\
        --layer_num 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cudf
import cupy as cp
import mlflow
# The cuML device selection is not needed here as this script only uses cudf/cupy
# from cuml.common.device_selection import get_global_device_type, set_global_device_type

# === MLflow setup ===
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply 'All-but-mean' normalization on GPU.")
    parser.add_argument("--source_path", type=Path, required=True, help="Path to the source Parquet file.")
    parser.add_argument("--out_path", type=Path, required=True, help="Path to save the normalized Parquet file.")
    parser.add_argument("--experiment_name", default="all-but-mean-normalization", help="Name for the MLflow experiment.")
    parser.add_argument("--layer_num", type=int, required=True, help="Layer number of the embeddings.")
    parser.add_argument("--provenance", type=str, help="Provenance information as a JSON string.")
    return parser.parse_args()


def is_file_valid(file_path: Path, min_size: int = 100) -> bool:
    """Check if a file exists and has a minimum size."""
    if not file_path.is_file():
        print(f"File not found: {file_path}")
        return False
    if file_path.stat().st_size < min_size:
        print(f"File {file_path} is too small ({file_path.stat().st_size} bytes), likely corrupt.")
        return False
    return True


def main():
    """Main function to run the normalization process."""
    args = parse_args()

    # The script uses cudf and cupy, which are GPU-native.
    # The successful import of these libraries is the guarantee of GPU execution.
    # The explicit device_type check is removed as it's not relevant for non-cuML operations.

    mlflow.set_experiment(args.experiment_name)
    run_name = f"normalize_layer_{args.layer_num}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(vars(args))
        provenance = json.loads(args.provenance) if args.provenance else {}
        mlflow.log_params(provenance)

        # Validate input file before processing
        if not is_file_valid(args.source_path):
            raise ValueError(f"Input file is missing or invalid: {args.source_path}")

        print("Loading data with cuDF...")
        gdf = cudf.read_parquet(args.source_path)
        
        # Store labels and drop the column to get the feature matrix
        labels = gdf['label']
        vectors_df = gdf.drop(columns=['label'])

        print(f"Data loaded. Shape: {vectors_df.shape}")
        
        # Convert to CuPy array for numerical operations
        vectors_cp = vectors_df.to_cupy()

        # Calculate global mean vector on GPU
        print("Calculating global mean vector...")
        global_mean = cp.mean(vectors_cp, axis=0)
        
        # All-but-mean normalization
        print("Applying 'all-but-mean' normalization IN-PLACE to save memory...")
        vectors_cp -= global_mean
        
        # Create a new DataFrame with normalized vectors and labels
        print("Reconstructing DataFrame...")
        normalized_df = cudf.DataFrame(vectors_cp, columns=vectors_df.columns)
        normalized_df['label'] = labels.reset_index(drop=True)

        # Explicitly free up GPU memory
        print("Freeing up memory...")
        del gdf
        del vectors_df
        del vectors_cp
        cp.get_default_memory_pool().free_all_blocks()

        print(f"Saving normalized data to {args.out_path}...")
        normalized_df.to_parquet(args.out_path)
        mlflow.log_artifact(str(args.out_path), artifact_path="normalized_embeddings")
        
        print("Done.")


if __name__ == "__main__":
    main() 