#!/usr/bin/env python
"""
experiments/02_umap_sklearn.py - UMAP using sklearn (CPU-based)
===============================================================
Uses sklearn UMAP instead of cuML to avoid GPU numerical instability issues.
This is more reliable for research purposes.

Usage:
    python experiments/02_umap_sklearn.py \\
        --pca_path data/snli/pca/pca_snli_50.parquet \\
        --out_path data/snli/umap/umap_snli_50_15_euclidean.parquet \\
        --n_neighbors 15 \\
        --metric euclidean \\
        --dataset snli \\
        --experiment_name umap-snli
"""
import argparse
import time
from pathlib import Path
import mlflow
import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
plt.style.use('seaborn-v0_8-darkgrid')
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

def parse_args():
    p = argparse.ArgumentParser(description="UMAP on PCA/ZCA reduced embeddings using sklearn (CPU).")
    p.add_argument("--pca_path", required=True, help="Path to Parquet file with PCA/ZCA vectors and labels.")
    p.add_argument("--out_path", required=True, help="Full path for the output UMAP Parquet file.")
    p.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP.")
    p.add_argument("--metric", default='euclidean', help="Metric for UMAP.")
    p.add_argument("--n_components", type=int, default=2, help="Number of components for UMAP (usually 2 for visualization).")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., 'snli', 'folio').")
    p.add_argument("--experiment_name", default="umap-roberta-base", help="Experiment name for MLflow.")
    p.add_argument("--reduction_type", choices=["pca", "zca"], required=True, help="Input reduction type.")
    p.add_argument("--layer_num", type=int, required=True, help="Layer number of source embeddings.")
    p.add_argument("--input_n_components", type=int, required=True, help="Number of input components (from PCA/ZCA).")
    p.add_argument("--skipped_n_components", type=int, default=0, help="Number of PCA/ZCA components skipped (for logging).")
    return p.parse_args()

def generate_scatter_plot(X, y, title, save_path):
    """Generates and saves a scatter plot."""
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label_val, color in zip(unique_labels, colors):
        idx = (y == label_val)
        plt.scatter(X[idx, 0], X[idx, 1], label=str(label_val), alpha=0.6, s=12, c=[color])
    
    plt.title(title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    if len(unique_labels) <= 10:  # Only show legend if not too many labels
        plt.legend(title="Label")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Scatter plot saved to {save_path}")

def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)

    run_name = f"umap_{args.dataset}_l{args.layer_num}_s{args.skipped_n_components}_{args.reduction_type}{args.input_n_components}_n{args.n_neighbors}_{args.metric}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Running UMAP (sklearn): {run_name} ---")
        mlflow.log_params(vars(args))
        mlflow.set_tag("compute_backend", "cpu_sklearn")
        mlflow.set_tag("reduction_step", "umap")

        print(f"Loading data from: {args.pca_path}")
        df = pd.read_parquet(args.pca_path)
        labels = df['label']
        X = df.drop('label', axis=1)

        print(f"Data loaded. Shape: {X.shape}")
        print(f"Running UMAP with n_neighbors={args.n_neighbors}, metric='{args.metric}'...")
        
        start_time = time.time()
        umap = UMAP(
            n_neighbors=args.n_neighbors,
            n_components=args.n_components,
            metric=args.metric,
            random_state=args.random_state,
            n_jobs=1  # Single-threaded for reproducibility
        )
        
        umap_results = umap.fit_transform(X.values)
        duration = time.time() - start_time
        
        print(f"UMAP processing completed in {duration:.2f} seconds.")
        mlflow.log_metric("umap_duration_sec", duration)

        # Check for problematic values
        nan_count = np.isnan(umap_results).sum()
        inf_count = np.isinf(umap_results).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"WARNING: Found {nan_count} NaN and {inf_count} Inf values in UMAP results")
            mlflow.log_metric("nan_values", nan_count)
            mlflow.log_metric("inf_values", inf_count)
            
            # Remove problematic rows
            valid_mask = ~(np.isnan(umap_results).any(axis=1) | np.isinf(umap_results).any(axis=1))
            umap_results = umap_results[valid_mask]
            labels = labels.iloc[valid_mask]
            
            print(f"Removed {(~valid_mask).sum()} problematic rows. Final shape: {umap_results.shape}")
        else:
            print("✅ No NaN/Inf values found in UMAP results")

        # Create output DataFrame
        umap_df = pd.DataFrame(
            umap_results, 
            columns=[f'umap_{i}' for i in range(umap_results.shape[1])],
            index=labels.index
        )
        final_df = pd.concat([umap_df, labels], axis=1)

        # Save results to Parquet
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(out_path)
        print(f"UMAP results saved to: {out_path}")
        mlflow.log_artifact(str(out_path), "umap_embeddings")

        # Log data quality metrics
        mlflow.log_metric("output_rows", len(final_df))
        mlflow.log_metric("umap_range_min", umap_results.min())
        mlflow.log_metric("umap_range_max", umap_results.max())
        mlflow.log_metric("umap_std", umap_results.std())

        # --- Generate and Save Scatter Plot ---
        print("Generating scatter plot...")
        plot_title = f"UMAP on {args.dataset} (Layer {args.layer_num}, {args.reduction_type.upper()} {args.input_n_components} dims)"
        scatter_path = out_path.with_suffix('.png')
        
        generate_scatter_plot(umap_results, labels.values, title=plot_title, save_path=scatter_path)
        mlflow.log_artifact(str(scatter_path), "plots")

    print("\n✅ UMAP processing completed.")

if __name__ == "__main__":
    main() 