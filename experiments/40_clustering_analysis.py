#!/usr/bin/env python3
"""
experiments/03_kmeans.py – GPU-Optimized KMeans Clustering
=========================================================
Pure cuML KMeans implementation with cluster evaluation and visualization.
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
from cuml.cluster import KMeans as GPU_KMeans
import mlflow

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimized KMeans clustering")
    parser.add_argument("--input_path", required=True, help="Input parquet with vectors and labels")
    parser.add_argument("--out_path", required=True, help="Output path for results JSON")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dataset_name", default="snli", help="Dataset name")
    parser.add_argument("--experiment_name", default="kmeans-gpu")
    parser.add_argument("--reduction_type", choices=["pca", "umap"], required=True)
    parser.add_argument("--layer_num", type=int, default=12)
    parser.add_argument("--input_n_components", type=str, required=False)
    parser.add_argument("--umap_n_neighbors", type=str, required=False)
    parser.add_argument("--umap_metric", type=str, required=False)
    parser.add_argument("--umap_source_reduction_type", type=str, choices=["pca", "zca"], required=False)
    parser.add_argument("--original_pca_n_components_before_slice", type=int, required=False)
    parser.add_argument("--skipped_n_components", type=int, required=False)
    parser.add_argument("--normalization_type", default="", help="Normalization method used")
    parser.add_argument("--config", default="", help="Configuration (EC/ECN)")
    parser.add_argument("--provenance", type=str, help="JSON string with pipeline parameters")
    parser.add_argument("--run_id", default="", help="MLflow run ID")
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

def compute_purity_gpu(y_true, y_pred):
    """Compute cluster purity using GPU operations"""
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    
    unique_true = cp.unique(y_true)
    unique_pred = cp.unique(y_pred)
    
    contingency_matrix = cp.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            contingency_matrix[i, j] = cp.sum((y_true == true_label) & (y_pred == pred_label))
    
    return cp.sum(cp.amax(contingency_matrix, axis=0)) / cp.sum(contingency_matrix)

def compute_nmi_gpu(y_true, y_pred):
    """Compute normalized mutual information using GPU operations"""
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    
    n = len(y_true)
    unique_true = cp.unique(y_true)
    unique_pred = cp.unique(y_pred)
    
    # Calculate entropies
    h_true = 0
    for label in unique_true:
        p = cp.sum(y_true == label) / n
        if p > 0:
            h_true -= p * cp.log2(p)
    
    h_pred = 0
    for label in unique_pred:
        p = cp.sum(y_pred == label) / n
        if p > 0:
            h_pred -= p * cp.log2(p)
    
    # Calculate mutual information
    mi = 0
    for true_label in unique_true:
        for pred_label in unique_pred:
            joint_count = cp.sum((y_true == true_label) & (y_pred == pred_label))
            if joint_count > 0:
                p_joint = joint_count / n
                p_true = cp.sum(y_true == true_label) / n
                p_pred = cp.sum(y_pred == pred_label) / n
                mi += p_joint * cp.log2(p_joint / (p_true * p_pred))
    
    return 2 * mi / (h_true + h_pred) if h_true > 0 and h_pred > 0 else 0.0

def create_cluster_plot(X, labels, predictions, output_path, k, purity, nmi):
    """Create and save cluster visualization plots"""
    try:
        # Convert to numpy for plotting
        if hasattr(X, 'values'):
            X_np = cp.asnumpy(X.values)
        else:
            X_np = cp.asnumpy(X)
        
        if hasattr(labels, 'values'):
            labels_np = cp.asnumpy(labels.values)
        else:
            labels_np = cp.asnumpy(labels)
        
        if hasattr(predictions, 'values'):
            pred_np = cp.asnumpy(predictions.values)
        else:
            pred_np = cp.asnumpy(predictions)
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: True labels
        unique_true = np.unique(labels_np)
        colors_true = ['red', 'blue', 'green', 'orange', 'purple']
        label_names = ['Entailment', 'Contradiction', 'Neutral']
        
        for i, label in enumerate(unique_true):
            mask = labels_np == label
            color = colors_true[i % len(colors_true)]
            label_name = label_names[i] if i < len(label_names) else f'Label {label}'
            
            ax1.scatter(X_np[mask, 0], X_np[mask, 1], 
                       c=color, alpha=0.6, s=2, label=label_name)
        
        ax1.set_title('True Labels')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predicted clusters
        unique_pred = np.unique(pred_np)
        colors_pred = ['purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for i, cluster in enumerate(unique_pred):
            mask = pred_np == cluster
            color = colors_pred[i % len(colors_pred)]
            
            ax2.scatter(X_np[mask, 0], X_np[mask, 1], 
                       c=color, alpha=0.6, s=2, label=f'Cluster {cluster}')
        
        ax2.set_title(f'K-Means Clusters (k={k})\nPurity: {purity:.3f}, NMI: {nmi:.3f}')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Cluster plot saved: {output_path}")
        mlflow.log_artifact(str(output_path))
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to create cluster plot: {e}")
        return False

def validate_input_data(df):
    """Validate and clean input data"""
    print(f"Input data shape: {df.shape}")
    
    # Check for required columns
    if 'label' not in df.columns:
        raise ValueError("Input data must contain 'label' column")
    
    # Identify feature columns (exclude label and ID columns)
    exclude_cols = ['label', 'premise_id', 'hypothesis_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")
    
    print(f"Feature columns: {len(feature_cols)} (excluded: {exclude_cols})")
    
    # Check for NaN/Inf values
    initial_rows = len(df)
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after cleaning NaN values")
    
    if initial_rows > len(df_clean):
        print(f"Dropped {initial_rows - len(df_clean)} rows with NaN values")
    
    # Check for Inf values
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

def run_kmeans_gpu(X, k, max_iter=300, random_state=42):
    """Run KMeans clustering with GPU optimization"""
    print(f"Running KMeans with k={k} on {len(X):,} samples")
    
    # Ensure data is float32 for GPU efficiency
    X = X.astype('float32')
    
    # Run KMeans
    kmeans = GPU_KMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=random_state,
        verbose=True
    )
    
    predictions = kmeans.fit_predict(X)
    
    # Validate predictions
    if cp.any(cp.isnan(predictions)):
        raise ValueError("KMeans produced NaN predictions")
    
    unique_clusters = cp.unique(predictions)
    print(f"Generated {len(unique_clusters)} unique clusters: {cp.asnumpy(unique_clusters)}")
    
    return predictions, kmeans

def process_kmeans_gpu(input_path: str, output_path: str, k: int, max_iter: int = 300, random_state: int = 42) -> dict:
    """Main KMeans processing function"""
    print(f"Loading data from {input_path}")
    
    # Load and validate data
    df = cudf.read_parquet(input_path)
    if df.empty:
        raise ValueError("Input data is empty")
    
    df_clean, feature_cols = validate_input_data(df)
    
    # Extract features and labels
    X = df_clean[feature_cols].values
    true_labels = df_clean['label'].values
    
    # Run KMeans
    predictions, kmeans_model = run_kmeans_gpu(X, k, max_iter, random_state)
    
    # Compute metrics
    print("Computing cluster evaluation metrics...")
    purity = compute_purity_gpu(true_labels, predictions)
    nmi = compute_nmi_gpu(true_labels, predictions)
    
    print(f"Cluster Purity: {purity:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    # Create visualization plot
    plot_path = Path(output_path).parent / f"kmeans_plot_{Path(output_path).stem}.png"
    create_cluster_plot(X, true_labels, predictions, plot_path, k, purity, nmi)
    
    # Save results
    results = {
        'k': k,
        'n_samples': len(df_clean),
        'n_features': len(feature_cols),
        'purity': float(purity),
        'nmi': float(nmi),
        'inertia': float(kmeans_model.inertia_),
        'n_iter': int(kmeans_model.n_iter_)
    }
    
    # Save results to JSON
    print(f"Saving results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    del X, predictions
    aggressive_cleanup()
    
    return results

def main():
    args = parse_args()
    
    # Handle MLflow run creation - Flat structure
    if hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    
    # Create run name with config first (if available)
    if hasattr(args, 'config') and args.config:
        run_name = f"{args.run_id}_{args.config}_layer_{args.layer_num}_40_clustering_k{args.k}" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_{args.config}_layer_{args.layer_num}_40_clustering_k{args.k}"
    else:
        run_name = f"{args.run_id}_layer_{args.layer_num}_40_clustering_k{args.k}" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_layer_{args.layer_num}_40_clustering_k{args.k}"
    
    with mlflow.start_run(run_name=run_name) as run:
        start_time = time.time()
        
        # Log provenance if provided (before args to avoid conflicts)
        if hasattr(args, 'provenance') and args.provenance:
            try:
                provenance = json.loads(args.provenance)
                # Filter out any parameters that might conflict with args
                filtered_provenance = {k: v for k, v in provenance.items() if not hasattr(args, k) or getattr(args, k) == ''}
                if filtered_provenance:
                    mlflow.log_params(filtered_provenance)
            except json.JSONDecodeError:
                print("Warning: Could not decode provenance JSON")
        
        # Log all parameters automatically
        mlflow.log_params(vars(args))
        
        # Set tags
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("dataset", args.dataset_name)
        mlflow.set_tag("layer_num", args.layer_num)
        mlflow.set_tag("reduction_type", args.reduction_type)
        
        try:
            # Execute KMeans
            results = process_kmeans_gpu(
                str(args.input_path),
                str(args.out_path),
                args.k,
                args.max_iter,
                args.random_state
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
            
            print(f"✓ KMeans completed successfully in {execution_time:.1f}s")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"✗ KMeans failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
