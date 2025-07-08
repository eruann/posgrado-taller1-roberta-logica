#!/usr/bin/env python
"""
scripts/consolidate_probe_results.py
====================================
This script gathers the results from all individual decision tree probe
experiments for a specific embedding type and consolidates them into a single, 
structured JSON file.

It supports two main sources:
1. Standard probes on PCA-reduced embeddings ('full', 'delta').
2. Probes on contrastive analysis embeddings ('contrastive_*').

This master file is ideal for programmatic analysis or for feeding into
a Large Language Model for interpretation.
"""
import json
import argparse
from pathlib import Path

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Consolidate decision tree probe results.")
    parser.add_argument(
        "--base_output_dir", 
        type=Path, 
        default=Path("results/probes"),
        help="Base directory where all probe results are stored. Should contain subdirs for 'full', 'delta', 'from_contrastive', etc."
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=[
            "full", "delta", 
            "contrastive_arithmetic_mean", 
            "contrastive_geometric_median", 
            "contrastive_cross_differences"
        ],
        help="The type of embedding results to consolidate."
    )
    return parser.parse_args()

def main():
    """
    Main function to find, parse, and consolidate probe results for a specific type.
    """
    args = parse_args()
    
    # --- Configuration from args ---
    mode = args.embedding_type
    layers = [9, 10, 11, 12]
    all_results = []
    
    # Define the final output file path. All consolidated files will be saved in the base output dir.
    output_file = args.base_output_dir / f"consolidated_results_{mode}.json"
    
    print(f"🚀 Starting consolidation for EMBEDDING_TYPE = {mode.upper()}...")

    if mode in ["full", "delta"]:
        # --- LOGIC FOR STANDARD PCA-BASED PROBES ---
        normalization_types = ["none", "per_type", "all_but_mean"]
        # Path for these is like: <base_output_dir>/full/dt_probe_pca_full_none_layer_9
        results_root = args.base_output_dir / mode
        print(f"Input directory tree: {results_root}")

        for norm_type in normalization_types:
            for layer in layers:
                exp_name = f"dt_probe_pca_{mode}_{norm_type}_layer_{layer}"
                exp_dir = results_root / exp_name
                
                summary_path = exp_dir / "probe_summary.json"
                rules_path = exp_dir / "tree_rules.txt"

                if not summary_path.exists() or not rules_path.exists():
                    print(f"⚠️  Skipping: Results not found for {exp_name}")
                    continue

                try:
                    with open(summary_path, 'r') as f:
                        summary_data = json.load(f)
                    with open(rules_path, 'r') as f:
                        rules_text = f.read()
                        
                    consolidated_record = {
                        "embedding_mode": mode,
                        "normalization_type": norm_type,
                        "layer": layer,
                        "metrics": summary_data.get("metrics", {}),
                        "top_features": summary_data.get("top_features", []),
                        "decision_rules": rules_text
                    }
                    all_results.append(consolidated_record)
                    print(f"✓ Consolidated: {exp_name}")

                except Exception as e:
                    print(f"✗ ERROR: Failed to process {exp_name}. Reason: {e}")

    elif mode.startswith("contrastive"):
        # --- LOGIC FOR NEW CONTRASTIVE-BASED PROBES ---
        method_name = mode.replace("contrastive_", "")
        # Path for these is like: <base_output_dir>/from_contrastive/arithmetic_mean/layer_9
        results_root = args.base_output_dir / "from_contrastive" / method_name
        print(f"Input directory tree: {results_root}")
        
        for layer in layers:
            exp_dir = results_root / f"layer_{layer}"
            
            summary_path = exp_dir / "probe_summary.json"
            rules_path = exp_dir / "tree_rules.txt"

            if not summary_path.exists() or not rules_path.exists():
                print(f"⚠️  Skipping: Results not found for layer {layer}")
                continue

            try:
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                with open(rules_path, 'r') as f:
                    rules_text = f.read()

                consolidated_record = {
                    "embedding_mode": "contrastive",
                    "contrastive_method": method_name,
                    "layer": layer,
                    "metrics": summary_data.get("metrics", {}),
                    "top_features": summary_data.get("top_features", []),
                    "decision_rules": rules_text
                }
                all_results.append(consolidated_record)
                print(f"✓ Consolidated: {method_name} layer {layer}")
            
            except Exception as e:
                print(f"✗ ERROR: Failed to process {exp_dir}. Reason: {e}")

    # Write the master list to the output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n🎉 Consolidation complete. {len(all_results)} experiment results saved to {output_file}")

if __name__ == "__main__":
    main() 