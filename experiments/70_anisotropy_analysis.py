#!/usr/bin/env python
"""
experiments/70_anisotropy_analysis.py - Análisis de Anisotropía Individual
========================================================================
Script individual para medir anisotropía de embeddings normalizados.

Usage:
    python experiments/70_anisotropy_analysis.py \
        --input_path data/snli/normalized/all_but_mean/normalized_snli_layer_9_all_but_mean.parquet \
        --output_dir data/snli/anisotropy/all_but_mean \
        --dataset snli \
        --layer_num 9 \
        --experiment_name anisotropy_analysis
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Individual anisotropy analysis for normalized embeddings")
    parser.add_argument("--input_path", required=True, type=Path, help="Path to normalized embeddings file")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory for results")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., snli, folio)")
    parser.add_argument("--layer_num", required=True, type=int, help="Layer number")
    parser.add_argument("--experiment_name", default="anisotropy_analysis", help="MLflow experiment name")
    parser.add_argument("--embedding_type", default="full", choices=["full", "delta"], 
                       help="Type of embedding (full or delta)")
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

def calculate_anisotropy(file_path: Path, emb_type: str, calculations: list) -> dict:
    """Invokes the anisotropy calculation script and returns the results."""
    print(f"  -> Calculating anisotropy for: {file_path.name}...")
    cmd = [
        "python", "scripts/utilities/calculate_anisotropy.py",
        "--input_path", str(file_path),
        "--embedding_type", emb_type,
        "--calculations", *calculations
    ]
    output = run_command(cmd)
    # Extract the JSON part from the output
    try:
        json_str = output.split("--- Results ---")[1].split("---------------")[0].strip()
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Could not parse JSON from anisotropy script output: {e}", file=sys.stderr)
        return {}

def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=f"anisotropy_{args.dataset}_layer{args.layer_num}") as run:
        print(f"--- Starting Anisotropy Analysis ---")
        print(f"Input: {args.input_path}")
        print(f"Output: {args.output_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Layer: {args.layer_num}")
        print(f"Embedding Type: {args.embedding_type}")
        
        # Log parameters
        mlflow.log_params({
            "input_path": str(args.input_path),
            "dataset": args.dataset,
            "layer_num": args.layer_num,
            "embedding_type": args.embedding_type
        })
        
        # Calculate anisotropy
        if args.input_path.exists():
            metrics = calculate_anisotropy(args.input_path, args.embedding_type, ["s_intra", "s_inter"])
            
            if metrics:
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Save results to file
                results_file = args.output_dir / f"anisotropy_{args.dataset}_layer{args.layer_num}.json"
                with open(results_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Log artifact
                mlflow.log_artifact(str(results_file))
                
                print(f"✅ Anisotropy analysis completed successfully!")
                print(f"Results saved to: {results_file}")
                print(f"Metrics: {metrics}")
            else:
                print(f"❌ Failed to calculate anisotropy metrics")
        else:
            print(f"❌ Input file not found: {args.input_path}")
            sys.exit(1)

if __name__ == "__main__":
    main() 