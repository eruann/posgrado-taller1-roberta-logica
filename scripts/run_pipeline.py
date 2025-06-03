#!/usr/bin/env python
"""
scripts/run_pipeline.py – Pipeline for running PCA, UMAP and KMeans experiments
=====================================================================
Runs a pipeline of experiments for each embedding layer from 9 to 12:
1. PCA with different dimensions (1, 5, 50)
2. UMAP on PCA outputs (with both 15 and 100 neighbors)
   - Using different metrics: cosine, euclidean, mahalanobis
3. KMeans on UMAP outputs

Each step is logged to MLflow with appropriate parameters and artifacts.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for running experiments")
    parser.add_argument("--data_dir", default="data/snli/embeddings",
                      help="Directory containing embedding files")
    parser.add_argument("--output_dir", default="data/snli/experiments",
                      help="Base directory for experiment outputs")
    return parser.parse_args()

def run_command(cmd):
    """Run a command and return its output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        raise Exception(f"Command failed: {' '.join(cmd)}")
    return result.stdout

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    pca_dir = output_dir / "pca"
    umap_dir = output_dir / "umap"
    kmeans_dir = output_dir / "kmeans"
    
    # Clean up existing directories
    for d in [pca_dir, umap_dir, kmeans_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # Define UMAP metrics to try
    umap_metrics = ["cosine", "euclidean", "mahalanobis"]
    
    # Process each layer
    for layer in range(9, 13):
        embedding_file = data_dir / f"embeddings_snli_layer_{layer}.parquet"
        if not embedding_file.exists():
            print(f"Warning: {embedding_file} not found, skipping...")
            continue
            
        print(f"\nProcessing layer {layer}...")
        
        # Run PCA for different dimensions
        for n_components in [1, 5, 50]:
            pca_output = pca_dir / f"pca_snli_{n_components}_layer{layer}.parquet"
            
            # Run PCA
            pca_cmd = [
                "python", "experiments/01_pca.py",
                "--source_path", str(embedding_file),
                "--out", str(pca_output),
                "--n_components", str(n_components),
                "--experiment_name", "pca",
                "--dataset", "snli",
                "--layer_num", str(layer)
            ]
            run_command(pca_cmd)
            
            # Run UMAP on PCA output with different neighbor values and metrics
            for n_neighbors in [15, 100]:
                for metric in umap_metrics:
                    # Update UMAP output filename to include metric
                    umap_output = umap_dir / f"umap_snli_{n_components}_layer{layer}_n{n_neighbors}_{metric}.parquet"
                    umap_cmd = [
                        "python", "experiments/02_umap.py",
                        "--pca_path", str(pca_output),
                        "--out_dir", str(umap_dir),
                        "--n_neighbors", str(n_neighbors),
                        "--min_dist", "0.1",
                        "--metric", metric,
                        "--dataset", "snli",
                        "--experiment_name", "umap",
                        "--layer_num", str(layer)
                    ]
                    run_command(umap_cmd)
                    
                    # Run KMeans on UMAP output with updated filename
                    kmeans_output = kmeans_dir / f"kmeans_snli_umap_{n_components}_layer{layer}_n{n_neighbors}_{metric}_k3.csv"
                    kmeans_cmd = [
                        "python", "experiments/03_kmeans.py",
                        "--input_path", str(umap_output),
                        "--out_dir", str(kmeans_dir),
                        "--k", "3",
                        "--dataset", "snli",
                        "--experiment_name", "kmeans",
                        "--reduction_type", "umap",
                        "--layer_num", str(layer)
                    ]
                    run_command(kmeans_cmd)

if __name__ == "__main__":
    main()