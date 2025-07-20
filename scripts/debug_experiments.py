#!/usr/bin/env python3
"""
Debug script to test MLflow experiment listing
"""

import mlflow
import mlflow.tracking
from pathlib import Path
from datetime import datetime, timedelta

def setup_mlflow_tracking():
    """Setup MLflow tracking URI."""
    current_dir = Path.cwd()
    mlruns_path = None
    
    for path in [current_dir] + list(current_dir.parents):
        potential_mlruns = path / "mlruns"
        if potential_mlruns.exists():
            mlruns_path = potential_mlruns
            break
    
    if mlruns_path:
        tracking_uri = f"file://{mlruns_path.absolute()}"
    else:
        tracking_uri = "file://./mlruns"
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

def main():
    setup_mlflow_tracking()
    
    client = mlflow.tracking.MlflowClient()
    
    print("Getting all experiments...")
    all_experiments = client.search_experiments()
    print(f"Found {len(all_experiments)} experiments")
    
    for exp in all_experiments[:10]:  # Limit to first 10
        print(f"- {exp.name} (ID: {exp.experiment_id})")
        
        # Get some runs for this experiment
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=5
            )
            print(f"  Runs: {len(runs)}")
            
            for run in runs[:3]:  # Show first 3 runs
                print(f"    - {run.info.run_name} ({run.info.run_id[:8]}...)")
                
                # Check for artifacts
                try:
                    artifacts = client.list_artifacts(run.info.run_id)
                    artifact_names = [art.path for art in artifacts]
                    print(f"      Artifacts: {artifact_names}")
                except Exception as e:
                    print(f"      Error listing artifacts: {e}")
                    
        except Exception as e:
            print(f"  Error getting runs: {e}")
        
        print()

if __name__ == "__main__":
    main() 