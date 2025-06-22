#!/usr/bin/env python
"""
Converts a single Parquet file from the 'narrow' format (with a 'vector' list column)
to the 'wide' format (with 'feature_i' columns for direct GPU loading).

This is intended as a one-time migration script to avoid re-running the full
embedding extraction pipeline.

Example:
    python scripts/experimental/convert_to_wide_format.py \\
        --source data/snli/embeddings_narrow/embeddings_snli_layer_12.parquet \\
        --dest data/snli/embeddings/embeddings_snli_layer_12.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cudf
import cupy as cp


def convert_to_wide(source_path: Path, dest_path: Path):
    """
    Converts a Parquet file from 'narrow' to 'wide' format using GPU.
    """
    print(f"Loading narrow file: {source_path}...")
    try:
        gdf = cudf.read_parquet(source_path)
    except Exception as e:
        print(f"Error reading {source_path}: {e}")
        return

    if "vector" not in gdf.columns:
        print(f"File {source_path} has no 'vector' column or is not in the expected narrow format. Skipping.")
        return

    labels = gdf["label"]

    print("Unpacking vector column to wide format on GPU...")
    n_samples = len(gdf)
    if n_samples == 0:
        print("File is empty. Creating an empty wide-format file.")
        wide_df = cudf.DataFrame()
        wide_df["label"] = labels
    else:
        # This is the GPU-native operation that was previously the bottleneck
        n_features = len(gdf["vector"].iloc[0])
        vectors_cp = gdf["vector"].list.leaves.to_cupy().reshape(n_samples, n_features)
        
        # Free memory explicitly
        del gdf
        cp.get_default_memory_pool().free_all_blocks()

        print("Creating wide DataFrame...")
        wide_df = cudf.DataFrame(vectors_cp, columns=[f"feature_{i}" for i in range(n_features)])
        wide_df["label"] = labels.reset_index(drop=True)

    print(f"Saving wide file to: {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move to pandas (CPU RAM) before writing to free up GPU memory
    wide_df_pandas = wide_df.to_pandas()
    
    # Explicitly free GPU memory now
    del wide_df
    del vectors_cp
    cp.get_default_memory_pool().free_all_blocks()
    
    wide_df_pandas.to_parquet(dest_path)
    print(f"✅ Conversion complete for {source_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert narrow embedding Parquet files to wide format.")
    parser.add_argument("--source", type=Path, required=True, help="Source narrow Parquet file.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination wide Parquet file.")
    args = parser.parse_args()

    convert_to_wide(args.source, args.dest)


if __name__ == "__main__":
    main() 