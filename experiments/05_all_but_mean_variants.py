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
    parser.add_argument("--provenance", default="{}", help="Provenance JSON string")
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
    """Extract premise, hypothesis, and delta embeddings from the composite dataframe"""
    try:
        # Get column groups
        premise_cols = [col for col in df.columns if col.startswith('premise_')]
        hypothesis_cols = [col for col in df.columns if col.startswith('hypothesis_')]
        delta_cols = [col for col in df.columns if col.startswith('delta_')]
        
        if not all([premise_cols, hypothesis_cols, delta_cols]):
            raise ValueError("Missing required embedding columns")
        
        print(f"Extracting embeddings: {len(premise_cols)} dimensions each")
        
        # Use cudf's values directly for efficiency
        premise_emb = cp.asarray(df[premise_cols].values, dtype=cp.float32)
        hypothesis_emb = cp.asarray(df[hypothesis_cols].values, dtype=cp.float32)
        delta_emb = cp.asarray(df[delta_cols].values, dtype=cp.float32)
        labels = df['label'].values
        
        return premise_emb, hypothesis_emb, delta_emb, labels
        
    except Exception as e:
        raise RuntimeError(f"Embedding extraction failed: {e}")

def apply_normalization(premise_emb, hypothesis_emb, delta_emb, norm_type):
    """Apply the specified normalization strategy using GPU operations"""
    print(f"Applying '{norm_type}' normalization...")
    
    try:
        if norm_type == 'all_but_mean':
            # Global mean across all vectors
            combined = cp.concatenate([premise_emb, hypothesis_emb, delta_emb], axis=0)
            global_mean = cp.mean(combined, axis=0)
            
            norm_premise = premise_emb - global_mean
            norm_hypothesis = hypothesis_emb - global_mean
            norm_delta = delta_emb - global_mean
            
            del combined, global_mean
            
        elif norm_type == 'per_type':
            # Separate mean for each vector type
            norm_premise = premise_emb - cp.mean(premise_emb, axis=0)
            norm_hypothesis = hypothesis_emb - cp.mean(hypothesis_emb, axis=0)
            norm_delta = delta_emb - cp.mean(delta_emb, axis=0)
        
        elif norm_type == 'standard':
            # Standard scaling (mean=0, std=1)
            premise_mean = cp.mean(premise_emb, axis=0)
            premise_std = cp.std(premise_emb, axis=0)
            premise_std = cp.where(premise_std < 1e-7, 1.0, premise_std)
            norm_premise = (premise_emb - premise_mean) / premise_std
            
            hypothesis_mean = cp.mean(hypothesis_emb, axis=0)
            hypothesis_std = cp.std(hypothesis_emb, axis=0)
            hypothesis_std = cp.where(hypothesis_std < 1e-7, 1.0, hypothesis_std)
            norm_hypothesis = (hypothesis_emb - hypothesis_mean) / hypothesis_std
            
            delta_mean = cp.mean(delta_emb, axis=0)
            delta_std = cp.std(delta_emb, axis=0)
            delta_std = cp.where(delta_std < 1e-7, 1.0, delta_std)
            norm_delta = (delta_emb - delta_mean) / delta_std
            
        elif norm_type == 'none':
            # Pass-through (no normalization)
            norm_premise = premise_emb.copy()
            norm_hypothesis = hypothesis_emb.copy()
            norm_delta = delta_emb.copy()
        
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        # Validate outputs
        for name, emb in [('premise', norm_premise), ('hypothesis', norm_hypothesis), ('delta', norm_delta)]:
            if cp.any(cp.isnan(emb)) or cp.any(cp.isinf(emb)):
                raise ValueError(f"Normalization produced NaN/Inf values in {name} embeddings")
        
        return norm_premise, norm_hypothesis, norm_delta
            
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

def save_normalized_embeddings(premise_emb, hypothesis_emb, delta_emb, labels, output_path, chunk_size=25000):
    """Save normalized embeddings in chunks for memory efficiency"""
    try:
        total_samples = len(labels)
        n_dims = premise_emb.shape[1]
        
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
        
        # Create column names
        premise_cols = [f'premise_{i}' for i in range(n_dims)]
        hypothesis_cols = [f'hypothesis_{i}' for i in range(n_dims)]
        delta_cols = [f'delta_{i}' for i in range(n_dims)]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        chunk_files = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Extract chunk data
            chunk_premise = premise_emb[start_idx:end_idx]
            chunk_hypothesis = hypothesis_emb[start_idx:end_idx]
            chunk_delta = delta_emb[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            # Create DataFrame
            chunk_data = {}
            
            # Add embeddings
            for i, col in enumerate(premise_cols):
                chunk_data[col] = cudf.Series(chunk_premise[:, i])
            for i, col in enumerate(hypothesis_cols):
                chunk_data[col] = cudf.Series(chunk_hypothesis[:, i])
            for i, col in enumerate(delta_cols):
                chunk_data[col] = cudf.Series(chunk_delta[:, i])
            
            chunk_data['label'] = cudf.Series(chunk_labels)
            
            chunk_df = cudf.DataFrame(chunk_data)
            
            # Save chunk
            chunk_file = output_path.parent / f"{output_path.stem}_chunk_{chunk_idx:04d}.parquet"
            chunk_df.to_parquet(chunk_file)
            chunk_files.append(chunk_file)
            
            # Progress update
            if (chunk_idx + 1) % 5 == 0 or chunk_idx == total_chunks - 1:
                print(f"  Saved chunk {chunk_idx + 1}/{total_chunks}")
            
            # Cleanup
            del chunk_data, chunk_df, chunk_premise, chunk_hypothesis, chunk_delta
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
    """Main normalization processing function"""
    print(f"Loading embeddings from {source_path}")
    
    # Load data
    df = cudf.read_parquet(source_path)
    if df.empty:
        raise ValueError("Input data is empty")
    
    print(f"Loaded {len(df):,} samples")
    
    # Extract embeddings
    premise_emb, hypothesis_emb, delta_emb, labels = extract_embeddings(df)
    
    # Compute pre-normalization stats
    print("Computing pre-normalization statistics...")
    pre_stats = {}
    pre_stats.update(compute_embedding_stats(premise_emb, "pre_premise"))
    pre_stats.update(compute_embedding_stats(hypothesis_emb, "pre_hypothesis"))
    pre_stats.update(compute_embedding_stats(delta_emb, "pre_delta"))
    
    # Apply normalization
    norm_premise, norm_hypothesis, norm_delta = apply_normalization(
        premise_emb, hypothesis_emb, delta_emb, normalization_type
    )
    
    # Compute post-normalization stats
    print("Computing post-normalization statistics...")
    post_stats = {}
    post_stats.update(compute_embedding_stats(norm_premise, "post_premise"))
    post_stats.update(compute_embedding_stats(norm_hypothesis, "post_hypothesis"))
    post_stats.update(compute_embedding_stats(norm_delta, "post_delta"))
    
    # Save normalized embeddings
    n_saved = save_normalized_embeddings(norm_premise, norm_hypothesis, norm_delta, labels, output_path)
    
    # Cleanup
    del premise_emb, hypothesis_emb, delta_emb
    del norm_premise, norm_hypothesis, norm_delta
    aggressive_cleanup()
    
    # Combine results
    results = {
        'normalization_type': normalization_type,
        'n_samples_processed': n_saved,
        'n_dimensions': premise_emb.shape[1] if 'premise_emb' in locals() else 768,
        **pre_stats,
        **post_stats
    }
    
    return results

def main():
    args = parse_args()
    
    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        start_time = time.time()
        
        # Log parameters
        mlflow.log_param("source_path", str(args.source_path))
        mlflow.log_param("out_path", str(args.out_path))
        mlflow.log_param("normalization_type", args.normalization_type)
        mlflow.log_param("layer_num", args.layer_num)
        
        # Log provenance
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