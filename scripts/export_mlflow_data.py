#!/usr/bin/env python
"""
scripts/export_mlflow_data.py
=============================
This script connects to the MLflow tracking server, fetches all experiment
runs, and dumps the data into a single, comprehensive CSV file.

The output includes run metadata, parameters, metrics, tags, and a list of
all associated artifact paths. This consolidated file is ideal for detailed
programmatic analysis, generating summary reports, or for input into a
Large Language Model for high-level interpretation.

Usage:
    # Export to the default 'mlflow_export.csv'
    python scripts/export_mlflow_data.py

    # Export to a custom file path
    python scripts/export_mlflow_data.py --output_path /path/to/my_analysis.csv

    # Specify a different tracking URI if not using a local 'mlruns' directory
    python scripts/export_mlflow_data.py --tracking_uri http://localhost:5000
"""
import argparse
import mlflow
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Export all MLflow data to a CSV file.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("mlflow_export.csv"),
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI. If not set, uses the default (e.g., local 'mlruns' directory)."
    )
    return parser.parse_args()

def main():
    """Main function to fetch and export MLflow data."""
    args = parse_args()
    
    print("🚀 Starting MLflow data export...")
    
    # --- 1. Connect to MLflow ---
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    print(f"🔍 Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # --- 2. Fetch all runs from all experiments ---
    # search_runs() is powerful. With search_all_experiments=True, it returns
    # a pandas DataFrame with params, metrics, and tags.
    try:
        runs_df = mlflow.search_runs(search_all_experiments=True)
        if runs_df.empty:
            print("❌ No runs found in MLflow. Exiting.")
            return
    except Exception as e:
        print(f"❌ Failed to fetch runs from MLflow. Error: {e}")
        print("   Ensure the tracking URI is correct and the server is accessible.")
        return

    print(f"✅ Found {len(runs_df)} total runs across all experiments.")

    # --- 3. Fetch artifact paths for each run ---
    # This is not included in search_runs, so we fetch it manually.
    print("⏳ Fetching artifact paths for each run (this may take a moment)...")
    
    artifact_paths = []
    client = mlflow.tracking.MlflowClient()
    
    for run_id in tqdm(runs_df["run_id"], desc="Processing artifacts"):
        try:
            artifacts = client.list_artifacts(run_id)
            # Join all artifact paths into a single string, separated by newlines
            paths_str = "\n".join([artifact.path for artifact in artifacts])
            artifact_paths.append(paths_str)
        except Exception:
            # Handle cases where a run might be deleted or has no artifacts
            artifact_paths.append(None)
            
    runs_df["artifacts"] = artifact_paths

    # --- 4. Clean up and save the data ---
    # Optional: Clean up column names by removing prefixes
    runs_df.columns = [
        col.replace("params.", "param_")
           .replace("metrics.", "metric_")
           .replace("tags.", "tag_")
        for col in runs_df.columns
    ]

    print(f"💾 Saving consolidated data to: {args.output_path}")
    
    try:
        runs_df.to_csv(args.output_path, index=False)
    except Exception as e:
        print(f"❌ Failed to save CSV file. Error: {e}")
        return

    print("\n🎉 Export complete!")
    print(f"📄 {len(runs_df)} records saved.")
    print(f"📈 {len([c for c in runs_df.columns if c.startswith('metric_')])} metrics, "
          f"{len([c for c in runs_df.columns if c.startswith('param_')])} parameters.")
    print(f"📍 File location: {args.output_path.resolve()}")

if __name__ == "__main__":
    main() 