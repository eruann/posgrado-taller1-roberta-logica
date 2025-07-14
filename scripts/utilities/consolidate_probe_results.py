#!/usr/bin/env python
"""
Scans the FOLIO analysis results directory to consolidate Decision Tree Probe
metrics (e.g., accuracy) from all experiments into a single CSV file.

This tool gathers scattered probe results by parsing the directory structure
where they are saved.
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Consolidate Decision Tree Probe results.")
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("data/folio/minimal_analysis/results"),
        help="Root directory of the analysis results."
    )
    p.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/folio/minimal_analysis/folio_probes_summary.csv"),
        help="Path to save the consolidated CSV summary."
    )
    return p.parse_args()

def parse_path_for_params(path: Path, base_dir: Path) -> dict:
    """
    Extracts experiment parameters from the directory path. This version is more
    flexible and handles different directory structures (like SNLI vs FOLIO)
    by searching for keywords instead of relying on fixed path positions.
    """
    params = {'normalization': 'none'} # Default value
    parts = path.relative_to(base_dir).parts
    
    for part in parts:
        part_lower = part.lower()
        
        # Identify the view
        if part_lower in ['full', 'delta'] or 'contrastive' in part_lower:
            params['view'] = 'contrastive' if 'contrastive' in part_lower else part_lower
        
        # Identify the layer
        elif 'layer' in part_lower:
            match = re.search(r'\d+', part_lower)
            if match:
                params['layer'] = int(match.group())
        
        # Identify normalization (if present)
        elif part_lower in ['none', 'all_but_mean', 'per_type']:
            params['normalization'] = part_lower

        # Identify dataset type for FOLIO
        elif part_lower in ['balanced', 'imbalanced']:
            params['dataset_type'] = part_lower
            
    return params

def main():
    """Main execution function."""
    args = parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found at '{args.results_dir}'")
        return

    print(f"🔍 Scanning for probe_summary.json files in '{args.results_dir}'...")
    
    json_files = list(args.results_dir.rglob("probe_summary.json"))
    
    if not json_files:
        print("No 'probe_summary.json' files found. Nothing to consolidate.")
        return
        
    all_results = []
    for file_path in json_files:
        params = parse_path_for_params(file_path, args.results_dir)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract accuracy from the nested metrics dictionary
            accuracy = data.get("metrics", {}).get("accuracy")
            
            if accuracy is not None:
                params['accuracy'] = accuracy
                all_results.append(params)
            else:
                print(f"⏩ Skipping {file_path} (missing accuracy in JSON)")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Could not read or parse {file_path}: {e}")
            continue
            
    if not all_results:
        print("Could not extract any valid results. No summary will be created.")
        return
        
    summary_df = pd.DataFrame(all_results)
    
    # Ensure a logical column order
    final_cols = ['dataset_type', 'view', 'layer', 'normalization', 'accuracy']
    summary_df = summary_df[[col for col in final_cols if col in summary_df.columns]]
    
    summary_df.sort_values(by=['view', 'layer', 'normalization'], inplace=True)
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_file, index=False)
    
    print("\n--- ✅ CONSOLIDATION COMPLETE ---")
    print("Summary of Decision Tree Probe Metrics:")
    print(summary_df.to_string())
    print(f"\nFull summary saved to: {args.output_file}")

if __name__ == "__main__":
    main() 