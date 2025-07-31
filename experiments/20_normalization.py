#!/usr/bin/env python3
"""
experiments/05_all_but_mean_variants.py
======================================
GPU-Optimized Normalization with Multiple Strategies
- 'all_but_mean': Global mean across all vectors
- 'per_type': Separate normalization for each vector type  
- 'standard': Standard scaling (mean=0, std=1) per type
- 'none': No normalization (pass-through)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import gc

import cudf
import cupy as cp
import numpy as np
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Apply normalization strategies to embeddings")
    parser.add_argument("--source_path", required=True, type=Path, help="Source embedding parquet file")
    parser.add_argument("--out_path", required=True, type=Path, help="Output normalized parquet file")
    parser.add_argument("--normalization_type", required=True, 
                       choices=['all_but_mean', 'per_type', 'standard', 'none'],
                       help="Normalization type")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment name")
    parser.add_argument("--layer_num", required=True, type=int, help="Layer number")
    parser.add_argument("--dataset_name", default="", help="Dataset name")
    parser.add_argument("--provenance", default="{}", help="Provenance JSON string")
    parser.add_argument("--run_id", default="", help="MLflow run ID")
    parser.add_argument("--config", default="", help="Configuration (EC/ECN)")
    return parser.parse_args()

def aggressive_cleanup():
    """Aggressive GPU memory cleanup"""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()

def log_scalar(key, value):
    """Log scalar values handling cupy and numpy types"""
    if hasattr(value, 'item'):
        value = value.item()
    elif isinstance(value, (cp.ndarray, np.ndarray)) and value.ndim == 0:
        value = float(value)
    mlflow.log_metric(key, value)

def extract_embeddings(df):
    """
    Extracts embeddings from the composite dataframe.
    Handles both 'full' (premise, hypothesis, delta) and 'single' (e.g., delta-only) structures.
    """
    try:
        # Get column groups, excluding ID columns from feature vectors
        premise_cols = [col for col in df.columns if col.startswith('premise_') and col != 'premise_id']
        hypothesis_cols = [col for col in df.columns if col.startswith('hypothesis_') and col != 'hypothesis_id']
        delta_cols = [col for col in df.columns if col.startswith('delta_')]

        # Identify non-feature columns to keep
        other_cols_to_keep = ['label']
        if 'premise_id' in df.columns:
            other_cols_to_keep.append('premise_id')
        if 'hypothesis_id' in df.columns:
            other_cols_to_keep.append('hypothesis_id')
        
        other_data = df[other_cols_to_keep]

        # Check for full structure
        if all([premise_cols, hypothesis_cols, delta_cols]):
            print("Extracting 'full' embeddings (premise, hypothesis, delta)...")
            premise_emb = cp.asarray(df[premise_cols].values, dtype=cp.float32)
            hypothesis_emb = cp.asarray(df[hypothesis_cols].values, dtype=cp.float32)
            delta_emb = cp.asarray(df[delta_cols].values, dtype=cp.float32)
            return (premise_emb, hypothesis_emb, delta_emb), other_data, 'full'

        # Check for single-vector structure (fallback)
        else:
            feature_cols = [col for col in df.columns if col not in other_cols_to_keep]
            if not feature_cols:
                 raise ValueError("No feature columns found in the dataframe.")
            
            print(f"Extracting 'single' vector type from {len(feature_cols)} feature columns...")
            embeddings = cp.asarray(df[feature_cols].values, dtype=cp.float32)
            return (embeddings,), other_data, 'single'
        
    except Exception as e:
        raise RuntimeError(f"Embedding extraction failed: {e}")

def apply_normalization(embeddings_tuple, norm_type):
    """Apply the specified normalization strategy using GPU operations"""
    print(f"Applying '{norm_type}' normalization...")
    
    is_full_structure = len(embeddings_tuple) == 3

    try:
        if is_full_structure:
            premise_emb, hypothesis_emb, delta_emb = embeddings_tuple
            
            if norm_type == 'all_but_mean':
                # Global mean across all vectors
                combined = cp.concatenate([premise_emb, hypothesis_emb, delta_emb], axis=0)
                global_mean = cp.mean(combined, axis=0)
                
                norm_premise = premise_emb - global_mean
                norm_hypothesis = hypothesis_emb - global_mean
                norm_delta = delta_emb - global_mean
                del combined
                
            elif norm_type == 'per_type':
                # Separate mean for each vector type
                norm_premise = premise_emb - cp.mean(premise_emb, axis=0)
                norm_hypothesis = hypothesis_emb - cp.mean(hypothesis_emb, axis=0)
                norm_delta = delta_emb - cp.mean(delta_emb, axis=0)
            
            elif norm_type == 'standard':
                # Standard scaling (mean=0, std=1)
                premise_mean, premise_std = cp.mean(premise_emb, axis=0), cp.std(premise_emb, axis=0)
                norm_premise = (premise_emb - premise_mean) / cp.where(premise_std < 1e-7, 1.0, premise_std)
                
                hypothesis_mean, hypothesis_std = cp.mean(hypothesis_emb, axis=0), cp.std(hypothesis_emb, axis=0)
                norm_hypothesis = (hypothesis_emb - hypothesis_mean) / cp.where(hypothesis_std < 1e-7, 1.0, hypothesis_std)
                
                delta_mean, delta_std = cp.mean(delta_emb, axis=0), cp.std(delta_emb, axis=0)
                norm_delta = (delta_emb - delta_mean) / cp.where(delta_std < 1e-7, 1.0, delta_std)
                
            elif norm_type == 'none':
                norm_premise, norm_hypothesis, norm_delta = premise_emb.copy(), hypothesis_emb.copy(), delta_emb.copy()
            
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")

            # Apply L2 normalization for relevant types
            if norm_type in ['all_but_mean', 'per_type']:
                for emb in [norm_premise, norm_hypothesis, norm_delta]:
                    norms = cp.linalg.norm(emb, axis=1, keepdims=True)
                    emb /= cp.where(norms < 1e-10, 1.0, norms)

            # Validate and return
            for emb in [norm_premise, norm_hypothesis, norm_delta]:
                if cp.any(cp.isnan(emb)) or cp.any(cp.isinf(emb)):
                    raise ValueError(f"Normalization produced NaN/Inf values")
            
            return (norm_premise, norm_hypothesis, norm_delta)

        # --- SINGLE VECTOR LOGIC ---
        else:
            embeddings, = embeddings_tuple
            
            if norm_type == 'all_but_mean' or norm_type == 'per_type':
                # For a single vector type, these are identical: just remove the mean
                print("   (Note: 'all_but_mean' and 'per_type' are equivalent to mean removal for single vector inputs)")
                norm_embeddings = embeddings - cp.mean(embeddings, axis=0)
                # And apply L2 norm
                norms = cp.linalg.norm(norm_embeddings, axis=1, keepdims=True)
                norm_embeddings /= cp.where(norms < 1e-10, 1.0, norms)

            elif norm_type == 'standard':
                mean, std = cp.mean(embeddings, axis=0), cp.std(embeddings, axis=0)
                norm_embeddings = (embeddings - mean) / cp.where(std < 1e-7, 1.0, std)
            
            elif norm_type == 'none':
                norm_embeddings = embeddings.copy()

            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")

            # Validate and return
            if cp.any(cp.isnan(norm_embeddings)) or cp.any(cp.isinf(norm_embeddings)):
                raise ValueError(f"Normalization produced NaN/Inf values")
                
            return (norm_embeddings,)

    except Exception as e:
        raise RuntimeError(f"Normalization failed: {e}")

def compute_embedding_stats(embeddings, name):
    """Compute embedding statistics for logging"""
    try:
        if isinstance(embeddings, np.ndarray):
            embeddings = cp.asarray(embeddings)
        
        # Basic statistics
        mean_norm = float(cp.linalg.norm(cp.mean(embeddings, axis=0)))
        std_norm = float(cp.std(cp.linalg.norm(embeddings, axis=1)))
        
        # Sample-based cosine similarity for efficiency
        sample_size = min(100, len(embeddings))
        sample_indices = cp.random.choice(len(embeddings), sample_size, replace=False)
        sample_emb = embeddings[sample_indices]
        
        cosine_sims = []
        for i in range(len(sample_emb)):
            for j in range(i+1, min(i+6, len(sample_emb))):  # Limit pairs for efficiency
                norm_i = cp.linalg.norm(sample_emb[i])
                norm_j = cp.linalg.norm(sample_emb[j])
                if norm_i > 1e-10 and norm_j > 1e-10:  # Avoid division by zero
                    sim = cp.dot(sample_emb[i], sample_emb[j]) / (norm_i * norm_j)
                    cosine_sims.append(float(sim))
        
        mean_cosine_sim = float(cp.mean(cp.array(cosine_sims))) if cosine_sims else 0.0
        
        return {
            f"{name}_mean_norm": mean_norm,
            f"{name}_std_norm": std_norm,
            f"{name}_mean_cosine_sim": mean_cosine_sim
        }
        
    except Exception as e:
        print(f"Warning: Failed to compute stats for {name}: {e}")
        return {}

def save_normalized_embeddings(embeddings_tuple, other_data_df, output_path, chunk_size=25000):
    """Save normalized embeddings in chunks for memory efficiency, handling both full and single structures."""
    try:
        total_samples = len(other_data_df)
        is_full_structure = len(embeddings_tuple) == 3
        
        if is_full_structure:
            premise_emb, hypothesis_emb, delta_emb = embeddings_tuple
            n_dims = premise_emb.shape[1]
            # Create column names for full structure
            premise_cols = [f'premise_{i}' for i in range(n_dims)]
            hypothesis_cols = [f'hypothesis_{i}' for i in range(n_dims)]
            delta_cols = [f'delta_{i}' for i in range(n_dims)]
        else:
            embeddings, = embeddings_tuple
            n_dims = embeddings.shape[1]
            # Create column names for single structure (assuming 'delta' for compatibility)
            feature_cols = [f'delta_{i}' for i in range(n_dims)]

        # Calculate memory requirements and adjust chunk size for 10GB GPU
        estimated_memory_per_sample = n_dims * 3 * 4 / (1024**3)  # 3 embeddings, float32
        estimated_chunk_memory = chunk_size * estimated_memory_per_sample
        
        # Conservative memory management for 10GB GPU
        if estimated_chunk_memory > 1.5:  # Use max 1.5GB per chunk
            chunk_size = int(1.5 / estimated_memory_per_sample)
            chunk_size = max(5000, min(chunk_size, 20000))
        
        total_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        print(f"Saving {total_samples:,} samples in {total_chunks} chunks of {chunk_size:,}")
        print(f"Estimated memory per chunk: {chunk_size * estimated_memory_per_sample:.2f} GB")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_files = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            chunk_data = {}

            if is_full_structure:
                # Handle full structure
                chunk_premise = premise_emb[start_idx:end_idx]
                chunk_hypothesis = hypothesis_emb[start_idx:end_idx]
                chunk_delta = delta_emb[start_idx:end_idx]
                for i, col in enumerate(premise_cols): chunk_data[col] = cudf.Series(chunk_premise[:, i])
                for i, col in enumerate(hypothesis_cols): chunk_data[col] = cudf.Series(chunk_hypothesis[:, i])
                for i, col in enumerate(delta_cols): chunk_data[col] = cudf.Series(chunk_delta[:, i])
            else:
                # Handle single structure
                chunk_embeddings = embeddings[start_idx:end_idx]
                for i, col in enumerate(feature_cols): chunk_data[col] = cudf.Series(chunk_embeddings[:, i])

            features_df = cudf.DataFrame(chunk_data)
            
            # Get the corresponding chunk of other data (labels, ids) and combine
            other_data_chunk = other_data_df.iloc[start_idx:end_idx].reset_index(drop=True)
            chunk_df = cudf.concat([features_df, other_data_chunk], axis=1)

            chunk_file = output_path.parent / f"{output_path.stem}_chunk_{chunk_idx:04d}.parquet"
            chunk_df.to_parquet(chunk_file)
            chunk_files.append(chunk_file)
            
            if (chunk_idx + 1) % 5 == 0 or chunk_idx == total_chunks - 1:
                print(f"  Saved chunk {chunk_idx + 1}/{total_chunks}")
            
            # Aggressive cleanup
            del features_df, other_data_chunk, chunk_df, chunk_data
            if is_full_structure: del chunk_premise, chunk_hypothesis, chunk_delta
            else: del chunk_embeddings
            aggressive_cleanup()
        
        # Combine chunks into final file using streaming approach
        print("Combining chunks using streaming approach...")
        
        try:
            # Use pandas for efficient streaming combination (avoids GPU memory issues)
            import pandas as pd
            
            # First, convert chunks to pandas and combine
            temp_output = output_path.with_suffix('.temp.parquet')
            
            pandas_chunks = []
            for i, chunk_file in enumerate(chunk_files):
                # Load chunk to GPU then immediately transfer to CPU
                gpu_chunk = cudf.read_parquet(chunk_file)
                cpu_chunk = gpu_chunk.to_pandas()
                pandas_chunks.append(cpu_chunk)
                
                # Clear GPU memory immediately
                del gpu_chunk
                aggressive_cleanup()
                
                if (i + 1) % 5 == 0 or i == len(chunk_files) - 1:
                    print(f"  Converted chunk {i + 1}/{len(chunk_files)} to CPU")
            
            # Combine on CPU
            print("Combining chunks on CPU...")
            combined_pandas = pd.concat(pandas_chunks, ignore_index=True)
            
            # Save directly from pandas (more memory efficient)
            combined_pandas.to_parquet(temp_output, engine='pyarrow')
            
            # Move temp file to final location
            temp_output.rename(output_path)
            
            # Get total rows before cleanup
            total_rows = len(combined_pandas)
            del pandas_chunks, combined_pandas
            
            print(f"✓ Saved normalized embeddings to {output_path}")
            return total_rows
            
        finally:
            # ALWAYS cleanup chunk files, even on error
            print("Cleaning up chunk files...")
            chunks_removed = 0
            for chunk_file in chunk_files:
                try:
                    if chunk_file.exists():
                        chunk_file.unlink()
                        chunks_removed += 1
                except Exception as e:
                    print(f"Warning: Could not remove chunk file {chunk_file}: {e}")
            
            print(f"Removed {chunks_removed}/{len(chunk_files)} chunk files")
            
            # Also cleanup temp file if it exists
            if 'temp_output' in locals() and temp_output.exists():
                try:
                    temp_output.unlink()
                    print("Removed temporary output file")
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_output}: {e}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save normalized embeddings: {e}")

def process_normalization_gpu(source_path: Path, output_path: Path, normalization_type: str) -> dict:
    """Main GPU-based normalization process"""
    start_time = time.time()
    
    # Load data
    print(f"Loading embeddings from {source_path}")
    df = cudf.read_parquet(source_path)
    print(f"Loaded {len(df):,} samples")
    
    # Extract embeddings
    embeddings_tuple, other_data, structure_type = extract_embeddings(df)
    
    # Compute stats for original embeddings
    if structure_type == 'full':
        original_stats = compute_embedding_stats(embeddings_tuple[0], "original_premise")
        original_stats.update(compute_embedding_stats(embeddings_tuple[1], "original_hypothesis"))
        original_stats.update(compute_embedding_stats(embeddings_tuple[2], "original_delta"))
    else:
        original_stats = compute_embedding_stats(embeddings_tuple[0], "original_vectors")

    # Apply normalization
    norm_embeddings_tuple = apply_normalization(embeddings_tuple, normalization_type)
    
    # Compute stats for normalized embeddings
    if structure_type == 'full':
        normalized_stats = compute_embedding_stats(norm_embeddings_tuple[0], "normalized_premise")
        normalized_stats.update(compute_embedding_stats(norm_embeddings_tuple[1], "normalized_hypothesis"))
        normalized_stats.update(compute_embedding_stats(norm_embeddings_tuple[2], "normalized_delta"))
    else:
        normalized_stats = compute_embedding_stats(norm_embeddings_tuple[0], "normalized_vectors")

    # Save results
    save_normalized_embeddings(norm_embeddings_tuple, other_data, output_path)
    
    # Final cleanup
    del df, embeddings_tuple, norm_embeddings_tuple, other_data
    aggressive_cleanup()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"✓ Normalization completed in {duration:.2f}s")
    
    # Combine stats for MLflow
    all_stats = {**original_stats, **normalized_stats, 'duration_s': duration}
    return all_stats

def main():
    args = parse_args()
    
    # Handle MLflow run creation - Flat structure
    if hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    
    # Create run name with config first (if available)
    if hasattr(args, 'config') and args.config:
        run_name = f"{args.run_id}_{args.config}_layer_{args.layer_num}_20_normalization_{args.normalization_type}" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_{args.config}_layer_{args.layer_num}_20_normalization_{args.normalization_type}"
    else:
        run_name = f"{args.run_id}_layer_{args.layer_num}_20_normalization_{args.normalization_type}" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_layer_{args.layer_num}_20_normalization_{args.normalization_type}"
    
    with mlflow.start_run(run_name=run_name) as run:
        start_time = time.time()
        
        # Log all parameters automatically
        mlflow.log_params(vars(args))
        
        # Log provenance if provided
        if hasattr(args, 'provenance') and args.provenance:
            try:
                provenance = json.loads(args.provenance)
                mlflow.log_params(provenance)
            except json.JSONDecodeError:
                print("Warning: Could not decode provenance JSON")
        
        # Set tags
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("layer_num", args.layer_num)
        mlflow.set_tag("normalization_type", args.normalization_type)
        
        try:
            # Execute normalization
            results = process_normalization_gpu(
                args.source_path,
                args.out_path,
                args.normalization_type
            )
            
            # Log results
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    log_scalar(key, value)
                else:
                    mlflow.log_param(key, str(value))
            
            # Log execution time
            execution_time = time.time() - start_time
            log_scalar("execution_time_seconds", execution_time)
            
            print(f"✓ Normalization completed successfully in {execution_time:.1f}s")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"✗ Normalization failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 