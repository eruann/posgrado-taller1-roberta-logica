#!/usr/bin/env python
"""
Runs a decision tree probe on a given set of embeddings.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import argparse
import json
import pickle
import mlflow
import pandas as pd
from experiments.probes.probe_utils import train_decision_tree_probe
from experiments.probes.analysis_utils import (
    get_classification_metrics,
    get_top_features,
    get_printable_rules,
)
from experiments.probes.visualization_utils import plot_tree_to_file

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a decision tree probe.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output files.")
    parser.add_argument("--experiment_name", type=str, default="decision_tree_probe", help="Name for the MLflow experiment.")
    parser.add_argument("--dataset_name", type=str, default="unknown", help="Name of the dataset being processed (e.g., 'folio', 'snli').")
    parser.add_argument("--embedding_type", type=str, required=True, help="Type of embedding (e.g., 'full', 'delta').")
    parser.add_argument("--layer_num", type=int, required=True, help="Layer number of the embeddings.")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum depth of the decision tree.")
    parser.add_argument("--min_samples_split", type=int, default=50, help="Min samples to split a node.")
    parser.add_argument("--scale_features", action='store_true', help="Apply StandardScaler to features before training.")
    parser.add_argument("--normalization_type", default="", help="Normalization method used")
    parser.add_argument("--provenance", type=str, default="{}", help="JSON string with provenance info from previous steps.")
    parser.add_argument("--run_id", type=str, default=None, help="MLflow run ID")
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle MLflow run creation - Flat structure
    if hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    
    # Create run with consistent naming pattern
    run_name = f"{args.run_id}_layer_{args.layer_num}_60_probing" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset_name}_layer_{args.layer_num}_60_probing"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Starting MLflow Run: {run.info.run_name} ---")
        
        # Log all parameters automatically
        mlflow.log_params(vars(args))
        
        # Log provenance if provided
        if hasattr(args, 'provenance') and args.provenance:
            try:
                provenance = json.loads(args.provenance)
                mlflow.log_params(provenance)
            except json.JSONDecodeError:
                print("Warning: Could not decode provenance JSON")

        # --- Train Probe ---
        probe_results = train_decision_tree_probe(
            data_path=args.input_path,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            scale_features=args.scale_features
        )
        
        # --- Analyze Results ---
        print("\n--- Analyzing Results ---")
        all_metrics = get_classification_metrics(probe_results["y_test"], probe_results["y_pred"])
        all_metrics["accuracy"] = probe_results["accuracy"]
        
        top_features_df = get_top_features(
            probe_results["feature_importances"],
            probe_results["feature_names"]
        )
        
        display_class_names = ["Entailment", "Contradiction"]

        tree_rules = get_printable_rules(
            probe_results["model"],
            probe_results["feature_names"],
            display_class_names,
            max_depth=3
        )
        
        # --- Display Summary ---
        print("\n--- PROBE RESULTS SUMMARY ---")
        print(f"Accuracy: {all_metrics['accuracy']:.4f}")
        print("\nTop 10 Most Informative Dimensions:")
        print(top_features_df.to_string(index=False))
        print("\n" + tree_rules)
        print("---------------------------\n")

        # --- Save Artifacts ---
        results_path = args.output_dir / "probe_summary.json"
        summary_data = {
            "metrics": all_metrics,
            "top_features": top_features_df.to_dict('records'),
        }
        with open(results_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        rules_path = args.output_dir / "tree_rules.txt"
        with open(rules_path, 'w') as f:
            f.write(tree_rules)
            
        model_path = args.output_dir / "decision_tree_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(probe_results["model"], f)
            
        plot_path = plot_tree_to_file(
            model=probe_results["model"],
            feature_names=probe_results["feature_names"],
            class_names=display_class_names,
            output_dir=args.output_dir,
            filename_prefix=run_name
        )

        # --- Log Artifacts to MLflow ---
        scalar_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}
        
        print("--- Logging artifacts to MLflow ---")
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(rules_path))
        mlflow.log_artifact(str(model_path))
        if plot_path and plot_path.exists():
            mlflow.log_artifact(str(plot_path))
            
        print(f"✓ Probe experiment completed successfully. Results in: {args.output_dir}")

if __name__ == "__main__":
    main() 