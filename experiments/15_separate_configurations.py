#!/usr/bin/env python3
"""
experiments/15_separate_configurations.py - Separate embeddings into EC and ECN configurations
==========================================================================
Separates embeddings into different configurations:
- EC: Only entailment (0) and contradiction (1) samples
- ECN: All samples including neutral (2)

Usage:
    python experiments/15_separate_configurations.py \
        --input_path data/embeddings.parquet \
        --output_path data/separated/EC/embeddings_EC.parquet \
        --config EC \
        --dataset snli \
        --layer_num 9
"""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Separate embeddings into EC/ECN configurations")
    parser.add_argument("--input_path", required=True, type=Path, help="Input embeddings file")
    parser.add_argument("--output_path", required=True, type=Path, help="Output separated file")
    parser.add_argument("--config", required=True, choices=["EC", "ECN"], help="Configuration to extract")
    parser.add_argument("--dataset_name", required=True, help="Dataset name")
    parser.add_argument("--layer_num", required=True, type=int, help="Layer number")
    parser.add_argument("--experiment_name", default="embeddings-separation")
    parser.add_argument("--run_id", default="", help="MLflow run ID")
    parser.add_argument("--provenance", default="{}", help="Provenance JSON string")
    return parser.parse_args()


def separate_embeddings(input_path: Path, output_path: Path, config: str, dataset_name: str):
    """Separate embeddings based on configuration."""
    print(f"📖 Reading embeddings from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"📊 Original dataset shape: {df.shape}")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Define label mappings for different datasets
    if dataset_name.lower() == "folio":
        # Folio: 'True' (entailment), 'False' (contradiction), 'Uncertain' (neutral)
        ec_labels = ['True', 'False']
        ecn_labels = ['True', 'False', 'Uncertain']
    else:
        # SNLI: 0 (entailment), 1 (contradiction), 2 (neutral)
        ec_labels = [0, 1]
        ecn_labels = [0, 1, 2]
    
    # Filter based on configuration
    if config == "EC":
        filtered_df = df[df['label'].isin(ec_labels)]
        print(f"🔀 EC configuration: keeping only labels {ec_labels}")
    elif config == "ECN":
        filtered_df = df[df['label'].isin(ecn_labels)]
        print(f"🔀 ECN configuration: keeping all labels {ecn_labels}")
    
    print(f"📊 Filtered dataset shape: {filtered_df.shape}")
    print(f"📊 Filtered label distribution:")
    print(filtered_df['label'].value_counts().sort_index())
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save filtered data
    filtered_df.to_parquet(output_path)
    print(f"💾 Saved {config} configuration to: {output_path}")
    
    return filtered_df


def main():
    args = parse_args()
    
    # Set up MLflow
    if hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    
    # Create run name with config first
    run_name = f"{args.run_id}_{args.config}_layer_{args.layer_num}_15_separation" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_{args.config}_layer_{args.layer_num}_15_separation"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log all parameters automatically
        mlflow.log_params(vars(args))
        
        # Log provenance if provided
        if hasattr(args, 'provenance') and args.provenance:
            try:
                provenance = json.loads(args.provenance)
                mlflow.log_params(provenance)
            except json.JSONDecodeError:
                print("Warning: Could not decode provenance JSON")
        
        # Separate embeddings
        filtered_df = separate_embeddings(args.input_path, args.output_path, args.config, args.dataset_name)
        
        # Log metrics
        mlflow.log_metric("original_samples", len(pd.read_parquet(args.input_path)))
        mlflow.log_metric("filtered_samples", len(filtered_df))
        mlflow.log_metric("reduction_ratio", len(filtered_df) / len(pd.read_parquet(args.input_path)))
        
        # Log label distribution
        label_counts = filtered_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            mlflow.log_metric(f"label_{label}_count", count)
        
        print(f"✅ Separation completed successfully!")
        print(f"📊 Configuration: {args.config}")
        print(f"📊 Output: {args.output_path}")


if __name__ == "__main__":
    main() 