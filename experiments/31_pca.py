#!/usr/bin/env python3
"""
experiments/01_pca.py – GPU-Optimized PCA for Large Datasets
============================================================
Pure cuDF/cuML implementation for 550k samples × 2304 features on 10GB GPU.
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
from cuml.decomposition import PCA as GPU_PCA
from cuml.decomposition import IncrementalPCA as GPU_IncrementalPCA
import mlflow

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def get_gpu_memory_gb():
    """Get available GPU memory in GB"""
    try:
        mempool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        total_memory = device.mem_info[1]  # Total memory in bytes
        return total_memory / (1024**3)  # Convert to GB
    except Exception as e:
        print(f"Warning: Could not detect GPU memory, defaulting to 8GB: {e}")
        return 8.0  # Conservative default

def calculate_optimal_chunk_size(n_features: int, gpu_memory_gb: float, mode: str = "medium") -> int:
    """Calculate optimal chunk size based on GPU memory and data dimensions"""
    
    # Memory utilization factors for different modes
    utilization_factors = {
        "small": 0.15,   # Very conservative - 15% of GPU memory
        "medium": 0.30,  # Balanced - 30% of GPU memory  
        "large": 0.50    # Aggressive - 50% of GPU memory
    }
    
    if mode not in utilization_factors:
        print(f"Warning: Unknown mode '{mode}', using 'medium'")
        mode = "medium"
    
    utilization_factor = utilization_factors[mode]
    
    # Memory overhead factor (accounts for intermediate computations, copies, etc.)
    memory_overhead = 4.0  # Conservative 4x overhead
    
    # Calculate chunk size
    # Formula: (GPU_memory * utilization) / (features * bytes_per_float * overhead)
    available_memory_bytes = gpu_memory_gb * utilization_factor * (1024**3)
    memory_per_sample = n_features * 4 * memory_overhead  # 4 bytes for float32
    
    optimal_chunk_size = int(available_memory_bytes / memory_per_sample)
    
    # Apply reasonable bounds
    min_chunk_size = 1000
    max_chunk_size = 100000
    
    chunk_size = max(min_chunk_size, min(max_chunk_size, optimal_chunk_size))
    
    print(f"GPU Memory: {gpu_memory_gb:.1f}GB, Mode: {mode}")
    print(f"Calculated chunk size: {chunk_size:,} samples")
    print(f"Estimated memory usage: {(chunk_size * memory_per_sample / (1024**3)):.2f}GB")
    
    return chunk_size

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimized PCA for large datasets")
    parser.add_argument("--source_path", required=True, help="Input parquet with vectors and 'label'")
    parser.add_argument("--out", required=True, help="Output parquet file path")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--chunk_size_mode", default="medium", choices=["small", "medium", "large"], 
                       help="Intelligent chunk sizing: small (conservative), medium (balanced), large (aggressive)")
    parser.add_argument("--experiment_name", default="pca-gpu")
    parser.add_argument("--dataset_name", required=True, help="Dataset name")
    parser.add_argument("--layer_num", type=int, required=True, help="Embedding layer number")
    parser.add_argument("--normalization_type", default="", help="Normalization method used")
    parser.add_argument("--config", default="", help="Configuration (EC/ECN)")
    parser.add_argument("--provenance", default="{}", help="Provenance JSON string")
    parser.add_argument("--run_id", default="", help="MLflow run ID")
    return parser.parse_args()

def log_scalar(key, value):
    """Log scalar values handling cupy and numpy types"""
    if hasattr(value, 'item'):
        value = value.item()
    elif isinstance(value, (cp.ndarray, np.ndarray)) and value.ndim == 0:
        value = float(value)
    mlflow.log_metric(key, value)

def aggressive_cleanup():
    """Aggressive GPU memory cleanup"""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()

def create_variance_plot(explained_variance_ratio, output_path):
    """Create and save explained variance plot"""
    try:
        # Convert to numpy
        if hasattr(explained_variance_ratio, 'values'):
            variance_ratio = cp.asnumpy(explained_variance_ratio.values)
        else:
            variance_ratio = cp.asnumpy(explained_variance_ratio)
        
        cumulative_variance = np.cumsum(variance_ratio)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Individual variance
        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio * 100)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Individual Variance Explained')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'bo-')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80%')
        ax2.axhline(y=95, color='g', linestyle='--', alpha=0.7, label='95%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Variance plot saved: {output_path}")
        mlflow.log_artifact(str(output_path))
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to create variance plot: {e}")
        return False

def process_pca_gpu(input_path: str, output_path: str, n_components: int, chunk_size_mode: str = "medium") -> dict:
    """Main PCA processing function with GPU optimization"""
    print(f"Loading data from {input_path}")
    
    # Load data
    df = cudf.read_parquet(input_path)
    if df.empty:
        raise ValueError("Input data is empty")
    
    # Identify feature columns (exclude label and ID columns)
    exclude_cols = ['label', 'premise_id', 'hypothesis_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"-> Feature columns: {len(feature_cols)} (excluded: {exclude_cols})")
    print(f"-> First few feature columns: {feature_cols[:5]}")
    print(f"-> Last few feature columns: {feature_cols[-5:]}")

    if not feature_cols:
        raise ValueError("Could not determine feature columns for PCA.")

    X_features = df[feature_cols]

    # Check for non-numeric dtypes before PCA
    if X_features.select_dtypes(include=['object', 'category']).shape[1] > 0:
        print("✗ PCA failed: Non-numeric columns found in feature set.", file=sys.stderr)
        sys.exit(1)

    print(f"Data shape: {X_features.shape[0]} samples × {X_features.shape[1]} features")

    if n_components >= X_features.shape[1]:
        print(f"Warning: n_components ({n_components}) is greater than or equal to the number of features ({X_features.shape[1]}). Setting n_components to {X_features.shape[1] - 1}.")
        n_components = X_features.shape[1] - 1
        print(f"New n_components: {n_components}")
    
    # Use intelligent chunk sizing
    gpu_memory_gb = get_gpu_memory_gb()
    chunk_size = calculate_optimal_chunk_size(X_features.shape[1], gpu_memory_gb, chunk_size_mode)
    
    # Memory estimation
    memory_needed_gb = (X_features.shape[0] * X_features.shape[1] * 4) / (1024**3)  # float32
    print(f"Estimated memory needed: {memory_needed_gb:.2f} GB")
    
    # Use incremental PCA for datasets > 3GB (conservative for 10GB GPU)
    use_incremental = memory_needed_gb > 3.0
    
    if use_incremental:
        print("Using Incremental PCA for large dataset")
        return _process_incremental_pca(df, feature_cols, output_path, n_components, chunk_size)
    else:
        print("Using Standard PCA for smaller dataset")
        return _process_standard_pca(df, feature_cols, output_path, n_components)

def _process_standard_pca(df, feature_cols, output_path, n_components):
    """Standard PCA for smaller datasets"""
    # Extract and normalize data
    X = df[feature_cols].values.astype('float32')
    
    # Input validation and cleaning
    if cp.any(cp.isnan(X)) or cp.any(cp.isinf(X)):
        print("Warning: NaN/Inf values detected, cleaning data")
        X = cp.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Normalize: center and scale
    print("Normalizing data...")
    mean_vals = cp.mean(X, axis=0)
    std_vals = cp.std(X, axis=0)
    std_vals = cp.where(std_vals < 1e-7, 1.0, std_vals)  # Avoid division by zero
    X_normalized = (X - mean_vals) / std_vals
    
    # Fit PCA with faster solver
    print(f"Fitting PCA with {n_components} components on {X_normalized.shape[0]} samples × {X_normalized.shape[1]} features...")
    pca = GPU_PCA(n_components=n_components, svd_solver='auto')
    X_pca = pca.fit_transform(X_normalized)
    print("PCA fitting completed!")
    
    # Output validation
    if cp.any(cp.isnan(X_pca)) or cp.any(cp.isinf(X_pca)):
        raise ValueError("PCA transformation produced NaN/Inf values")
    
    # Create output DataFrames
    pca_df = cudf.DataFrame({
        f'PCA_{i+1}': X_pca[:, i].astype('float32') for i in range(n_components)
    })
    # Preserve label as numeric codes (do NOT feed into PCA)
    if df['label'].dtype.kind in ('b', 'i', 'u', 'f'):
        pca_df['label'] = df['label'].astype('int8').values
        zca_label = df['label'].astype('int8').values
    else:
        codes = df['label'].astype('category').cat.codes
        pca_df['label'] = codes.astype('int8').values
        zca_label = codes.astype('int8').values
    
    # Create ZCA (whitened) variant
    eigenvalues = pca.explained_variance_
    eigenvalues = cp.where(eigenvalues < 1e-10, 1e-10, eigenvalues)
    whitening_matrix = pca.components_.T * cp.sqrt(1.0 / eigenvalues)
    X_zca = cp.dot(X_pca, whitening_matrix.T)
    
    zca_df = cudf.DataFrame({
        f'PCA_{i+1}': X_zca[:, i].astype('float32') for i in range(n_components)
    })
    zca_df['label'] = zca_label
    
    # Save both variants
    base_path = Path(output_path)
    if base_path.name.startswith("pca_"):
        # Caller already provided a pca_ prefixed filename
        pca_path = base_path
        zca_path = base_path.with_name(base_path.name.replace("pca_", "zca_", 1))
    else:
        # Add prefixes ourselves
        pca_path = base_path.parent / f"pca_{base_path.name}"
        zca_path = base_path.parent / f"zca_{base_path.name}"
    
    print(f"Saving PCA results to {pca_path}")
    pca_df.to_parquet(pca_path)
    
    print(f"Saving ZCA results to {zca_path}")
    zca_df.to_parquet(zca_path)
    
    # Create variance plots for both variants
    pca_plot_path = base_path.parent / f"pca_variance_{base_path.stem}.png"
    zca_plot_path = base_path.parent / f"zca_variance_{base_path.stem}.png"
    create_variance_plot(pca.explained_variance_ratio_, pca_plot_path)
    create_variance_plot(pca.explained_variance_ratio_, zca_plot_path)
    
    # Cleanup
    del X, X_normalized, X_pca, X_zca
    aggressive_cleanup()
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_explained_variance': float(cp.sum(pca.explained_variance_ratio_)),
        'n_components_used': n_components,
        'pca_method': 'standard_pca',
        'variants_created': ['pca', 'zca'],
        'pca_path': str(pca_path),
        'zca_path': str(zca_path)
    }

def _process_incremental_pca(df, feature_cols, output_path, n_components, chunk_size):
    """Incremental PCA for large datasets with chunked processing"""
    n_samples = len(df)
    n_features = len(feature_cols)
    
    print(f"Processing {n_samples:,} samples in chunks of {chunk_size:,}")
    
    # First pass: compute global statistics for normalization
    print("Computing global statistics...")
    mean_vals = cp.zeros(n_features, dtype='float64')
    var_vals = cp.zeros(n_features, dtype='float64')
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_data = df.iloc[start_idx:end_idx][feature_cols].values.astype('float64')
        
        chunk_mean = cp.mean(chunk_data, axis=0)
        chunk_var = cp.var(chunk_data, axis=0)
        chunk_size_actual = end_idx - start_idx
        
        mean_vals += chunk_mean * chunk_size_actual
        var_vals += chunk_var * chunk_size_actual
        
        del chunk_data
        aggressive_cleanup()
        
        if start_idx % (chunk_size * 10) == 0:
            print(f"  Statistics progress: {end_idx:,}/{n_samples:,}")
    
    mean_vals /= n_samples
    var_vals /= n_samples
    std_vals = cp.sqrt(var_vals)
    std_vals = cp.where(std_vals < 1e-7, 1.0, std_vals)
    
    print("Global statistics computed")
    
    # Second pass: fit incremental PCA
    print("Fitting Incremental PCA...")
    pca = GPU_IncrementalPCA(n_components=n_components, batch_size=chunk_size)
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_data = df.iloc[start_idx:end_idx][feature_cols].values.astype('float64')
        
        # Normalize chunk
        chunk_normalized = (chunk_data - mean_vals) / std_vals
        
        # Partial fit
        pca.partial_fit(chunk_normalized)
        
        del chunk_data, chunk_normalized
        aggressive_cleanup()
        
        if start_idx % (chunk_size * 10) == 0:
            print(f"  PCA fitting progress: {end_idx:,}/{n_samples:,}")
    
    print("Incremental PCA fitted")
    
    # Third pass: transform data for PCA variant
    print("Transforming data for PCA variant...")
    pca_chunks = []
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_data = df.iloc[start_idx:end_idx][feature_cols].values.astype('float64')
        
        # Normalize and transform
        chunk_normalized = (chunk_data - mean_vals) / std_vals
        chunk_pca = pca.transform(chunk_normalized)
        
        # Create DataFrame chunk
        chunk_df = cudf.DataFrame({
            f'PCA_{i+1}': chunk_pca[:, i].astype('float32') for i in range(n_components)
        })
        chunk_df['label'] = df.iloc[start_idx:end_idx]['label'].values
        
        pca_chunks.append(chunk_df)
        
        del chunk_data, chunk_normalized, chunk_pca
        aggressive_cleanup()
        
        if start_idx % (chunk_size * 10) == 0:
            print(f"  PCA transform progress: {end_idx:,}/{n_samples:,}")
    
    # Combine PCA results
    print("Combining PCA results...")
    pca_df = cudf.concat(pca_chunks, ignore_index=True)
    del pca_chunks
    aggressive_cleanup()
    
    # Fourth pass: create ZCA variant
    print("Creating ZCA variant...")
    eigenvalues = pca.explained_variance_
    eigenvalues = cp.where(eigenvalues < 1e-10, 1e-10, eigenvalues)
    whitening_matrix = pca.components_.T * cp.sqrt(1.0 / eigenvalues)
    
    zca_chunks = []
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_data = df.iloc[start_idx:end_idx][feature_cols].values.astype('float64')
        
        # Normalize and transform to PCA space
        chunk_normalized = (chunk_data - mean_vals) / std_vals
        chunk_pca = pca.transform(chunk_normalized)
        
        # Apply whitening transformation for ZCA
        chunk_zca = cp.dot(chunk_pca, whitening_matrix.T)
        
        # Create DataFrame chunk
        chunk_df = cudf.DataFrame({
            f'PCA_{i+1}': chunk_zca[:, i].astype('float32') for i in range(n_components)
        })
        chunk_df['label'] = df.iloc[start_idx:end_idx]['label'].values
        
        zca_chunks.append(chunk_df)
        
        del chunk_data, chunk_normalized, chunk_pca, chunk_zca
        aggressive_cleanup()
        
        if start_idx % (chunk_size * 10) == 0:
            print(f"  ZCA transform progress: {end_idx:,}/{n_samples:,}")
    
    # Combine ZCA results
    print("Combining ZCA results...")
    zca_df = cudf.concat(zca_chunks, ignore_index=True)
    del zca_chunks
    aggressive_cleanup()
    
    # Save both variants
    base_path = Path(output_path)
    if base_path.name.startswith("pca_"):
        # Caller already provided a pca_ prefixed filename
        pca_path = base_path
        zca_path = base_path.with_name(base_path.name.replace("pca_", "zca_", 1))
    else:
        # Add prefixes ourselves
        pca_path = base_path.parent / f"pca_{base_path.name}"
        zca_path = base_path.parent / f"zca_{base_path.name}"
    
    print(f"Saving PCA results to {pca_path}")
    pca_df.to_parquet(pca_path)
    
    print(f"Saving ZCA results to {zca_path}")
    zca_df.to_parquet(zca_path)
    
    # Create variance plots for both variants
    pca_plot_path = base_path.parent / f"pca_variance_{base_path.stem}.png"
    zca_plot_path = base_path.parent / f"zca_variance_{base_path.stem}.png"
    create_variance_plot(pca.explained_variance_ratio_, pca_plot_path)
    create_variance_plot(pca.explained_variance_ratio_, zca_plot_path)
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_explained_variance': float(cp.sum(pca.explained_variance_ratio_)),
        'n_components_used': n_components,
        'pca_method': 'incremental_pca',
        'variants_created': ['pca', 'zca'],
        'pca_path': str(pca_path),
        'zca_path': str(zca_path)
    }

def main():
    args = parse_args()
    
    # Handle MLflow run creation - Flat structure
    if hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    
    # Create run name with config first (if available)
    if hasattr(args, 'config') and args.config:
        run_name = f"{args.run_id}_{args.config}_layer_{args.layer_num}_31_pca_n{args.n_components}" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_{args.config}_layer_{args.layer_num}_31_pca_n{args.n_components}"
    else:
        run_name = f"{args.run_id}_layer_{args.layer_num}_31_pca_n{args.n_components}" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_layer_{args.layer_num}_31_pca_n{args.n_components}"
    
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
        mlflow.set_tag("dataset", args.dataset_name)
        mlflow.set_tag("layer_num", args.layer_num)
        
        try:
            # Execute PCA
            results = process_pca_gpu(
                str(args.source_path), 
                str(args.out), 
                args.n_components,
                args.chunk_size_mode
            )
            
            # Log results
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    log_scalar(key, value)
                elif isinstance(value, list) and len(value) <= 100:  # Don't log huge arrays
                    mlflow.log_param(key, str(value)[:500])  # Truncate if too long
                else:
                    mlflow.log_param(key, str(value))
            
            # Log execution time
            execution_time = time.time() - start_time
            log_scalar("execution_time_seconds", execution_time)
            
            print(f"✓ PCA completed successfully in {execution_time:.1f}s")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"✗ PCA failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 