#!/usr/bin/env python3
"""
experiments/02_umap.py - GPU-Optimized UMAP
==========================================
Pure cuML UMAP implementation with robust error handling and visualization.
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
import cuml
import mlflow

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimized UMAP")
    parser.add_argument("--pca_path", required=True, help="Input parquet with PCA/ZCA vectors and labels")
    parser.add_argument("--out_path", required=True, help="Output parquet file path")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP")
    parser.add_argument("--metric", default='euclidean', help="Distance metric")
    parser.add_argument("--n_components", type=int, default=2, help="Number of UMAP components")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--experiment_name", default="umap-gpu")
    parser.add_argument("--reduction_type", choices=["pca", "zca"], required=True)
    parser.add_argument("--layer_num", type=int, required=True)
    parser.add_argument("--input_n_components", type=int, required=True)
    parser.add_argument("--skipped_n_components", type=int, default=0)
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

def create_umap_plot(df, output_path):
    """Create and save UMAP visualization"""
    try:
        # Convert to numpy for plotting
        x_vals = cp.asnumpy(df['UMAP_1'].values)
        y_vals = cp.asnumpy(df['UMAP_2'].values)
        labels = cp.asnumpy(df['label'].values)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot by label with different colors
        unique_labels = np.unique(labels)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(x_vals[mask], y_vals[mask], 
                       c=colors[i % len(colors)], 
                       label=f'Label {label}', 
                       alpha=0.6, s=2)
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Projection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 UMAP plot saved: {output_path}")
        mlflow.log_artifact(str(output_path))
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to create UMAP plot: {e}")
        return False

def validate_input_data(df):
    """Validate and clean input data"""
    print(f"Input data shape: {df.shape}")
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col != 'label']
    print(f"Feature columns: {len(feature_cols)}")
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found (expected columns other than 'label')")
    
    # Check for NaN/Inf values
    total_nan = 0
    total_inf = 0
    
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        inf_count = cp.isinf(df[col].values).sum()
        total_nan += nan_count
        total_inf += inf_count
        
        if nan_count > 0 or inf_count > 0:
            print(f"Column {col}: {nan_count} NaN, {inf_count} Inf values")
    
    print(f"Total: {total_nan} NaN, {total_inf} Inf values")
    
    # Clean data by removing rows with NaN/Inf
    df_clean = df.dropna()  # GPU-optimized dropna
    
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after cleaning NaN values")
    
    # Additional check for Inf values
    X = df_clean[feature_cols].values
    if cp.any(cp.isinf(X)):
        print("Removing Inf values...")
        inf_mask = cp.any(cp.isinf(X), axis=1)
        valid_indices = cp.where(~inf_mask)[0]
        df_clean = df_clean.iloc[cp.asnumpy(valid_indices)]
        
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after cleaning Inf values")
    
    print(f"After cleaning: {df_clean.shape}")
    return df_clean, feature_cols

def run_umap_gpu(X, n_neighbors, min_dist, metric, n_components=2):
    """Run UMAP with GPU optimization and fallback handling"""
    n_samples = len(X)
    print(f"Running UMAP on {n_samples:,} samples with {X.shape[1]} features")
    
    # Adjust neighbors if too large
    n_neighbors = min(n_neighbors, n_samples - 1)
    print(f"Using {n_neighbors} neighbors")
    
    # Primary UMAP parameters
    umap_params = {
        'n_neighbors': n_neighbors,
        'n_components': n_components,
        'min_dist': min_dist,
        'metric': metric,
        'random_state': 42,
        'verbose': True,
        'low_memory': True,
        'n_epochs': 200,
        'learning_rate': 1.0
    }
    
    try:
        print("Attempting UMAP with primary parameters...")
        umap_model = cuml.UMAP(**umap_params)
        X_umap = umap_model.fit_transform(X)
        
        # Validate output
        if cp.any(cp.isnan(X_umap)) or cp.any(cp.isinf(X_umap)):
            raise ValueError("UMAP produced NaN/Inf values")
        
        print("✓ UMAP completed successfully")
        return X_umap
        
    except Exception as e:
        print(f"Primary UMAP failed: {e}")
        print("Trying fallback parameters...")
        
        # Fallback parameters
        fallback_params = {
            'n_neighbors': min(15, n_samples - 1),
            'n_components': n_components,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'random_state': 42,
            'verbose': False,
            'low_memory': True,
            'n_epochs': 100,
            'learning_rate': 0.5
        }
        
        try:
            umap_model = cuml.UMAP(**fallback_params)
            X_umap = umap_model.fit_transform(X)
            
            if cp.any(cp.isnan(X_umap)) or cp.any(cp.isinf(X_umap)):
                raise ValueError("Fallback UMAP also produced NaN/Inf values")
            
            print("✓ Fallback UMAP completed successfully")
            return X_umap
            
        except Exception as e2:
            raise RuntimeError(f"Both primary and fallback UMAP failed. Primary: {e}, Fallback: {e2}")

def process_umap_gpu(input_path: str, output_path: str, n_neighbors: int, min_dist: float, metric: str, n_components: int = 2) -> dict:
    """Main UMAP processing function"""
    print(f"Loading data from {input_path}")
    
    # Load and validate data
    df = cudf.read_parquet(input_path)
    if df.empty:
        raise ValueError("Input data is empty")
    
    df_clean, feature_cols = validate_input_data(df)
    
    # Extract features for UMAP
    X = df_clean[feature_cols].values.astype('float32')
    
    # Input normalization (optional - UMAP is fairly robust)
    print("Normalizing input data...")
    X_mean = cp.mean(X, axis=0)
    X_std = cp.std(X, axis=0)
    X_std = cp.where(X_std < 1e-7, 1.0, X_std)  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std
    
    # Run UMAP
    X_umap = run_umap_gpu(X_normalized, n_neighbors, min_dist, metric, n_components)
    
    # Create output DataFrame
    output_df = cudf.DataFrame({
        f'UMAP_{i+1}': X_umap[:, i].astype('float32') for i in range(n_components)
    })
    output_df['label'] = df_clean['label'].values
    
    # Final validation
    if len(output_df) == 0:
        raise ValueError("No data in final output")
    
    # Save results
    print(f"Saving UMAP results to {output_path}")
    output_df.to_parquet(output_path)
    
    # Create visualization plot
    if n_components == 2:
        plot_path = Path(output_path).parent / f"umap_plot_{Path(output_path).stem}.png"
        create_umap_plot(output_df, plot_path)
    
    # Cleanup
    del X, X_normalized, X_umap
    aggressive_cleanup()
    
    return {
        'n_samples_processed': len(output_df),
        'n_features_input': len(feature_cols),
        'n_components_output': n_components,
        'n_neighbors_used': n_neighbors,
        'min_dist_used': min_dist,
        'metric_used': metric
    }

def main():
    args = parse_args()
    
    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)
    
    # Create descriptive run name
    run_name = f"{args.dataset}_{args.n_neighbors}neighbors_{args.metric}_layer{args.layer_num}"
    
    with mlflow.start_run(run_name=run_name):
        start_time = time.time()
        
        # Log parameters
        mlflow.log_param("pca_path", str(args.pca_path))
        mlflow.log_param("out_path", str(args.out_path))
        mlflow.log_param("n_neighbors", args.n_neighbors)
        mlflow.log_param("min_dist", args.min_dist)
        mlflow.log_param("metric", args.metric)
        mlflow.log_param("n_components", args.n_components)
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("reduction_type", args.reduction_type)
        mlflow.log_param("layer_num", args.layer_num)
        mlflow.log_param("input_n_components", args.input_n_components)
        mlflow.log_param("skipped_n_components", args.skipped_n_components)
        
        # Log provenance
        try:
            provenance = json.loads(args.provenance)
            mlflow.log_params(provenance)
        except json.JSONDecodeError:
            print("Warning: Could not decode provenance JSON")
        
        # Set tags
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("dataset", args.dataset)
        mlflow.set_tag("layer_num", args.layer_num)
        mlflow.set_tag("reduction_type", args.reduction_type)
        
        try:
            # Execute UMAP
            results = process_umap_gpu(
                str(args.pca_path),
                str(args.out_path),
                args.n_neighbors,
                args.min_dist,
                args.metric,
                args.n_components
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
            
            print(f"✓ UMAP completed successfully in {execution_time:.1f}s")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"✗ UMAP failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
