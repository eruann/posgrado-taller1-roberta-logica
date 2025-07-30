#!/usr/bin/env python3
"""
experiments/30_dimensionality_reduction.py - Dimensionality Reduction Pipeline
============================================================================
Orchestrates PCA and UMAP dimensionality reduction steps.

This pipeline:
1. Applies PCA to reduce dimensions from high-dimensional embeddings to ~50 components
2. Applies UMAP to further reduce to 2D for visualization and clustering

Usage:
    python experiments/30_dimensionality_reduction.py \
        --source_path data/snli/embeddings/embeddings_snli_layer_12.parquet \
        --output_dir data/snli/dimensionality_reduction \
        --pca_components 50 \
        --umap_neighbors 15 \
        --umap_min_dist 0.1 \
        --umap_metric euclidean
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Dimensionality reduction pipeline")
    parser.add_argument("--source_path", required=True, type=Path, help="Input embedding file")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter")
    parser.add_argument("--umap_metric", default="euclidean", help="UMAP distance metric")
    parser.add_argument("--experiment_name", default="dimensionality_reduction")
    parser.add_argument("--dataset", required=True, help="Dataset name (snli, folio)")
    parser.add_argument("--layer_num", type=int, required=True, help="Layer number")
    parser.add_argument("--reduction_type", default="pca", choices=["pca", "zca"], help="Reduction type")
    parser.add_argument("--provenance", default="{}", help="Provenance JSON string")
    return parser.parse_args()

def run_command(cmd: list, cwd: Path = None) -> str:
    """Executes a command and returns its stdout, raising an error on failure."""
    print(f"\n▶️ RUNNING: {' '.join(map(str, cmd))}")
    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        print("✅ Command successful.")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ COMMAND FAILED: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(f"--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("--------------", file=sys.stderr)
        raise e

def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=f"dim_reduction_{args.dataset}_layer{args.layer_num}") as run:
        print(f"--- Starting Dimensionality Reduction Pipeline ---")
        
        # Log parameters
        mlflow.log_params({
            "source_path": str(args.source_path),
            "dataset": args.dataset,
            "layer_num": args.layer_num,
            "pca_components": args.pca_components,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_metric": args.umap_metric,
            "reduction_type": args.reduction_type
        })
        
        # Step 1: PCA
        print(f"\n📊 Step 1: PCA Reduction to {args.pca_components} components")
        pca_output = args.output_dir / f"pca_{args.dataset}_layer{args.layer_num}.parquet"
        
        pca_cmd = [
            "python", "experiments/31_pca.py",
            "--source_path", str(args.source_path),
            "--out", str(pca_output),
            "--n_components", str(args.pca_components),
            "--experiment_name", f"{args.experiment_name}_pca",
            "--dataset", args.dataset,
            "--layer_num", str(args.layer_num),
            "--provenance", args.provenance
        ]
        
        run_command(pca_cmd)
        
        # Step 2: UMAP
        print(f"\n🗺️ Step 2: UMAP Reduction to 2D")
        umap_output = args.output_dir / f"umap_{args.dataset}_layer{args.layer_num}.parquet"
        
        umap_cmd = [
            "python", "experiments/32_umap.py",
            "--pca_path", str(pca_output),
            "--out_path", str(umap_output),
            "--n_neighbors", str(args.umap_neighbors),
            "--min_dist", str(args.umap_min_dist),
            "--metric", args.umap_metric,
            "--n_components", "2",
            "--dataset", args.dataset,
            "--experiment_name", f"{args.experiment_name}_umap",
            "--reduction_type", args.reduction_type,
            "--layer_num", str(args.layer_num),
            "--input_n_components", str(args.pca_components),
            "--skipped_n_components", "0",
            "--provenance", args.provenance
        ]
        
        run_command(umap_cmd)
        
        # Log artifacts
        mlflow.log_artifact(str(pca_output))
        mlflow.log_artifact(str(umap_output))
        
        print(f"\n✅ Dimensionality reduction pipeline completed!")
        print(f"📁 PCA output: {pca_output}")
        print(f"📁 UMAP output: {umap_output}")

if __name__ == "__main__":
    main() 