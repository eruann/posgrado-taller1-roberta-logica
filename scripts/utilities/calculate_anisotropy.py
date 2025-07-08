#!/usr/bin/env python
"""
Calculates embedding anisotropy metrics (S_intra and S_inter) for a given
Parquet file containing vector embeddings.

Metrics:
- S_intra: Average cosine similarity between vectors of the same premise.
           Measures how focused the inference vectors are for a given context.
           Requires a 'premise_id' or 'premise_hash' column.
- S_inter: Average cosine similarity between vectors from different premises.
           Measures the overall spread or collapse of the embedding space.

Usage:
    python scripts/utilities/calculate_anisotropy.py \\
        --input_path /path/to/embeddings.parquet \\
        --embedding_type delta \\
        --calculations s_intra s_inter \\
        --sample_size 50000
"""

import argparse
import json
import sys

import cudf
import cupy as cp
from tqdm import tqdm

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Calculate embedding anisotropy metrics.")
    p.add_argument("--input_path", type=str, required=True, help="Path to the input Parquet file.")
    p.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["full", "delta", "contrastive"],
        help="Type of embeddings to guide column selection."
    )
    p.add_argument(
        "--calculations",
        nargs='+',
        required=True,
        choices=["s_intra", "s_inter"],
        help="List of calculations to perform."
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Number of vectors to sample for S_inter calculation."
    )
    return p.parse_args()

def get_feature_vectors(df: cudf.DataFrame, embedding_type: str) -> cp.ndarray:
    """
    Extracts the relevant feature vectors from a DataFrame based on the embedding type.
    """
    print(f"Extracting vectors for embedding type: '{embedding_type}'...")
    
    if embedding_type == 'full':
        # For 'full' concatenated embeddings, we are interested in the 'delta' part.
        # Assumes structure: [premise (768), hypothesis (768), delta (768)]
        delta_start_col = 768 + 768
        delta_end_col = delta_start_col + 768
        feature_cols = df.columns[delta_start_col:delta_end_col]
        print(f"-> Selected delta vector columns from index {delta_start_col} to {delta_end_col}.")
    elif embedding_type == 'delta':
        # For delta-only files, look for standard feature columns.
        feature_cols = [c for c in df.columns if c.startswith('feature_') or c.startswith('delta_')]
        print(f"-> Selected {len(feature_cols)} feature columns (delta-only).")
    elif embedding_type == 'contrastive':
        # For contrastive files, also look for standard feature columns.
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        print(f"-> Selected {len(feature_cols)} feature columns (contrastive).")
    
    if len(feature_cols) == 0:
        raise ValueError("Could not find any feature columns to process.")
        
    return cp.asarray(df[feature_cols].values, dtype=cp.float32)

def calculate_s_inter(vectors: cp.ndarray, sample_size: int, chunk_size: int = 1024) -> float:
    """
    Calculates the average inter-premise cosine similarity (S_inter) using a
    chunking strategy to avoid out-of-memory errors on the GPU.
    """
    print(f"Calculating S_inter using a sample of {sample_size} vectors...")
    if len(vectors) > sample_size:
        indices = cp.random.choice(len(vectors), sample_size, replace=False)
        sample = vectors[indices]
    else:
        sample = vectors
        print("-> Sample size is larger than dataset, using all vectors.")

    # L2 normalize the vectors
    norms = cp.linalg.norm(sample, axis=1, keepdims=True)
    sample_norm = sample / cp.where(norms < 1e-10, 1.0, norms)
    
    n_vectors = len(sample_norm)
    total_sim = cp.array(0.0, dtype=cp.float64)
    total_pairs = 0
    
    print(f"-> Processing in chunks of {chunk_size} to conserve memory...")
    for i in tqdm(range(0, n_vectors, chunk_size), desc="  -> S_inter chunks"):
        chunk = sample_norm[i : i + chunk_size]
        
        # 1. Similarity of the chunk with all vectors that come AFTER it
        if i + chunk_size < n_vectors:
            rest = sample_norm[i + chunk_size:]
            sim_block = chunk @ rest.T
            total_sim += cp.sum(sim_block)
            total_pairs += sim_block.size
            
        # 2. Similarity within the chunk itself (upper triangle)
        sim_intra_chunk = chunk @ chunk.T
        indices = cp.triu_indices(len(chunk), k=1)
        total_sim += cp.sum(sim_intra_chunk[indices])
        total_pairs += len(indices[0])

    if total_pairs == 0:
        return 0.0

    mean_sim = float(total_sim / total_pairs)
    print(f"-> S_inter result: {mean_sim:.6f}")
    return mean_sim

def calculate_s_intra(df: cudf.DataFrame, vectors: cp.ndarray) -> float:
    """
    Calculates the average intra-premise cosine similarity (S_intra).
    """
    group_col = None
    if 'premise_id' in df.columns:
        group_col = 'premise_id'
    elif 'premise_hash' in df.columns:
        group_col = 'premise_hash'
    
    if not group_col:
        print("-> S_intra requires 'premise_id' or 'premise_hash' column. Skipping.")
        return None

    print(f"Calculating S_intra by grouping on '{group_col}'...")
    
    # Using pandas for groupby-apply is more straightforward for this kind of operation.
    pdf = df[[group_col]].to_pandas()
    grouped = pdf.groupby(group_col)
    
    group_means = []
    # Use tqdm for a progress bar as this can be slow
    for _, group in tqdm(grouped, desc="  -> Processing groups", total=len(grouped)):
        if len(group) < 2:
            continue
        
        # Get vectors for the current group using original indices
        group_vectors = vectors[group.index.values]
        
        # L2 normalize
        norms = cp.linalg.norm(group_vectors, axis=1, keepdims=True)
        group_norm = group_vectors / cp.where(norms < 1e-10, 1.0, norms)
        
        # Calculate similarity matrix for the group
        sim_matrix = group_norm @ group_norm.T
        indices = cp.triu_indices(len(group), k=1)
        
        # Append the mean similarity of this group
        group_means.append(cp.mean(sim_matrix[indices]))
        
    if not group_means:
        print("-> No groups with 2 or more members found for S_intra calculation.")
        return None
        
    # Calculate the final S_intra as the mean of all group means
    mean_of_means = float(cp.mean(cp.array(group_means)))
    print(f"-> S_intra result: {mean_of_means:.6f} (from {len(group_means)} groups)")
    return mean_of_means

def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        print(f"Loading data from: {args.input_path}")
        df = cudf.read_parquet(args.input_path)
        print(f"Loaded {len(df):,} records.")
        
        vectors = get_feature_vectors(df, args.embedding_type)
        
        results = {}
        
        if "s_inter" in args.calculations:
            s_inter_val = calculate_s_inter(vectors, args.sample_size)
            if s_inter_val is not None:
                results["s_inter"] = s_inter_val

        if "s_intra" in args.calculations:
            s_intra_val = calculate_s_intra(df, vectors)
            if s_intra_val is not None:
                results["s_intra"] = s_intra_val

        # Print results as a JSON string to stdout for the pipeline to capture
        print("\n--- Results ---")
        print(json.dumps(results))
        print("---------------")

    except FileNotFoundError:
        print(f"FATAL: Input file not found at {args.input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()