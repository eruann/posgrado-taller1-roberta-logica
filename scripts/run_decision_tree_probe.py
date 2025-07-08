#!/usr/bin/env python
"""
scripts/run_decision_tree_probe.py
==================================
Main pipeline script to run a decision tree probe experiment.
This script orchestrates the training, analysis, and visualization of a
decision tree classifier to probe for informative dimensions in embeddings.
"""
import argparse
import json
import pickle
from pathlib import Path

import mlflow
from experiments.probes.probe_utils import train_decision_tree_probe
from experiments.probes.analysis_utils import (
    get_classification_metrics,
    get_top_features,
    get_printable_rules,
)
from experiments.probes.visualization_utils import plot_tree_to_file

def parse_args():
    parser = argparse.ArgumentParser(description="Run a Decision Tree Probe on embeddings.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the input Parquet file with embeddings.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save results and artifacts.")
    parser.add_argument("--experiment_name", type=str, default="DecisionTree_Probe", help="Name for the MLflow experiment.")
    parser.add_argument("--embedding_type", type=str, required=True, help="Type of embedding (e.g., 'full', 'delta', 'pca_full').")
    parser.add_argument("--layer_num", type=int, required=True, help="RoBERTa layer number for the embeddings.")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth of the decision tree.")
    parser.add_argument("--min_samples_split", type=int, default=50, help="Min samples to split a node.")
    parser.add_argument("--scale_features", action='store_true', help="Apply StandardScaler to features before training.")
    parser.add_argument("--provenance", type=str, default="{}", help="JSON string with provenance info from previous steps.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- MLflow Setup ---
    mlflow.set_experiment(args.experiment_name)
    run_name = f"probe_{args.embedding_type}_layer{args.layer_num}_depth{args.max_depth}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Starting MLflow Run: {run.info.run_name} ---")
        
        # Log parameters
        mlflow.log_params(vars(args))
        try:
            provenance = json.loads(args.provenance)
            mlflow.log_params({f"prov_{k}": v for k, v in provenance.items()})
        except json.JSONDecodeError:
            print("Warning: Could not parse provenance string.")

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
        
        # Define the human-readable class names for reporting.
        # This corrects the internal [0, 1] remapping back to meaningful labels.
        display_class_names = ["Entailment", "Contradiction"]

        tree_rules = get_printable_rules(
            probe_results["model"],
            probe_results["feature_names"],
            display_class_names, # Use corrected names
            max_depth=3
        )
        
        # --- Display Summary ---
        print("\n--- PROBE RESULTS SUMMARY ---")
        print(f"Accuracy: {all_metrics['accuracy']:.4f}")
        print(f"Precision: {all_metrics['precision']:.4f}")
        print(f"Recall: {all_metrics['recall']:.4f}")
        print("\nTop 10 Most Informative Dimensions:")
        print(top_features_df.to_string(index=False))
        print("\n" + tree_rules)
        print("---------------------------\n")

        # --- Save Artifacts ---
        # 1. Save metrics and results to a JSON file
        results_path = args.output_dir / "probe_summary.json"
        summary_data = {
            "metrics": all_metrics,
            "top_features": top_features_df.to_dict('records'),
        }
        with open(results_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        # 2. Save tree rules to a text file
        rules_path = args.output_dir / "tree_rules.txt"
        with open(rules_path, 'w') as f:
            f.write(tree_rules)
            
        # 3. Save the trained model
        model_path = args.output_dir / "decision_tree_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(probe_results["model"], f)
            
        # 4. Generate and save the tree visualization
        plot_path = plot_tree_to_file(
            model=probe_results["model"],
            feature_names=probe_results["feature_names"],
            class_names=display_class_names, # Use corrected names
            output_dir=args.output_dir,
            filename_prefix=run_name
        )

        # --- Log Artifacts to MLflow ---
        # Separate scalar metrics for logging from complex artifacts like the confusion matrix
        scalar_metrics = {k: v for k, v in all_metrics.items() if not isinstance(v, list)}

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