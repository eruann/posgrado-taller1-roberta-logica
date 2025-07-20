#!/usr/bin/env python3
"""
MLflow Run Information Retriever

This script connects to MLflow to retrieve experiment information
given a run_id. It provides both command-line and programmatic interfaces.

Usage:
    python get_experiment_info.py <run_id>
    
Or import as module:
    from get_experiment_info import get_run_info
    info = get_run_info("your-run-id-here")
"""

import mlflow
import mlflow.tracking
import sys
import os
from typing import Dict, Any, Optional
import argparse
from pathlib import Path


def setup_mlflow_tracking(tracking_uri: Optional[str] = None) -> None:
    """
    Setup MLflow tracking URI. 
    
    Args:
        tracking_uri: Custom tracking URI. If None, uses local mlruns directory
    """
    if tracking_uri is None:
        # Look for mlruns directory in current or parent directories
        current_dir = Path.cwd()
        mlruns_path = None
        
        # Check current directory and parents
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
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")


def get_run_info(run_id: str, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve comprehensive information about an MLflow run.
    
    Args:
        run_id: The MLflow run ID to query
        tracking_uri: Optional custom tracking URI
        
    Returns:
        Dictionary containing run information
        
    Raises:
        Exception: If run_id is not found or MLflow connection fails
    """
    setup_mlflow_tracking(tracking_uri)
    
    try:
        # Get the MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Get run information
        run = client.get_run(run_id)
        
        # Get experiment information
        experiment = client.get_experiment(run.info.experiment_id)
        
        # Get artifacts list
        artifacts = []
        try:
            artifacts_list = client.list_artifacts(run_id)
            for artifact in artifacts_list:
                artifact_info = {
                    'path': artifact.path,
                    'is_dir': artifact.is_dir,
                    'file_size': artifact.file_size
                }
                if artifact.is_dir:
                    # If it's a directory, list its contents
                    try:
                        sub_artifacts = client.list_artifacts(run_id, artifact.path)
                        artifact_info['contents'] = [
                            {
                                'path': sub_art.path,
                                'is_dir': sub_art.is_dir,
                                'file_size': sub_art.file_size
                            }
                            for sub_art in sub_artifacts
                        ]
                    except Exception:
                        artifact_info['contents'] = []
                artifacts.append(artifact_info)
        except Exception as e:
            print(f"Warning: Could not list artifacts: {e}")
            artifacts = []
        
        # Extract key information
        info = {
            'run_id': run.info.run_id,
            'run_name': run.info.run_name,
            'experiment_id': run.info.experiment_id,
            'experiment_name': experiment.name,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,
            'lifecycle_stage': run.info.lifecycle_stage,
            'user_id': run.info.user_id,
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'tags': dict(run.data.tags),
            'artifacts': artifacts
        }
        
        return info
        
    except Exception as e:
        raise Exception(f"Failed to retrieve run information for {run_id}: {str(e)}")


def print_run_info(run_info: Dict[str, Any], verbose: bool = False) -> None:
    """
    Print run information in a formatted way.
    
    Args:
        run_info: Dictionary containing run information
        verbose: Whether to print detailed information
    """
    print(f"\n{'='*60}")
    print(f"MLflow Run Information")
    print(f"{'='*60}")
    
    # Basic information
    print(f"Run ID:           {run_info['run_id']}")
    print(f"Run Name:         {run_info['run_name']}")
    print(f"Experiment ID:    {run_info['experiment_id']}")
    print(f"Experiment Name:  {run_info['experiment_name']}")
    print(f"Status:           {run_info['status']}")
    print(f"User:             {run_info['user_id']}")
    
    # Timestamps
    if run_info['start_time']:
        from datetime import datetime
        start_time = datetime.fromtimestamp(run_info['start_time'] / 1000)
        print(f"Start Time:       {start_time}")
    
    if run_info['end_time']:
        end_time = datetime.fromtimestamp(run_info['end_time'] / 1000)
        print(f"End Time:         {end_time}")
        
        if run_info['start_time']:
            duration = (run_info['end_time'] - run_info['start_time']) / 1000
            print(f"Duration:         {duration:.2f} seconds")
    
    if verbose:
        # Parameters
        if run_info['params']:
            print(f"\nParameters ({len(run_info['params'])}):")
            for key, value in sorted(run_info['params'].items()):
                print(f"  {key}: {value}")
        
        # Metrics
        if run_info['metrics']:
            print(f"\nMetrics ({len(run_info['metrics'])}):")
            for key, value in sorted(run_info['metrics'].items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        # Tags
        if run_info['tags']:
            print(f"\nTags ({len(run_info['tags'])}):")
            for key, value in sorted(run_info['tags'].items()):
                print(f"  {key}: {value}")
        
        print(f"\nArtifact URI:     {run_info['artifact_uri']}")
        
        # Artifacts
        if run_info['artifacts']:
            print(f"\nArtifacts ({len(run_info['artifacts'])}):")
            for artifact in run_info['artifacts']:
                if artifact['is_dir']:
                    print(f"  📁 {artifact['path']}/")
                    if 'contents' in artifact and artifact['contents']:
                        for sub_artifact in artifact['contents'][:10]:  # Limit to first 10 items
                            size_str = f" ({sub_artifact['file_size']} bytes)" if sub_artifact['file_size'] else ""
                            icon = "📁" if sub_artifact['is_dir'] else "📄"
                            print(f"    {icon} {sub_artifact['path']}{size_str}")
                        if len(artifact['contents']) > 10:
                            print(f"    ... and {len(artifact['contents']) - 10} more items")
                else:
                    size_str = f" ({artifact['file_size']} bytes)" if artifact['file_size'] else ""
                    print(f"  📄 {artifact['path']}{size_str}")


def main():
    """Command-line interface for the script."""
    parser = argparse.ArgumentParser(
        description="Retrieve MLflow run information by run_id",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_experiment_info.py f167bda8068942378dec32b2b0dce486
  python get_experiment_info.py f167bda8068942378dec32b2b0dce486 --verbose
  python get_experiment_info.py f167bda8068942378dec32b2b0dce486 --tracking-uri file:///path/to/mlruns
        """
    )
    
    parser.add_argument(
        'run_id',
        help='MLflow run ID to query'
    )
    
    parser.add_argument(
        '--tracking-uri',
        help='MLflow tracking URI (default: auto-detect local mlruns)',
        default=None
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information including parameters, metrics, and tags'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output information as JSON'
    )
    
    parser.add_argument(
        '--artifacts-only',
        action='store_true',
        help='Show only artifacts information'
    )
    
    args = parser.parse_args()
    
    try:
        # Get run information
        run_info = get_run_info(args.run_id, args.tracking_uri)
        
        if args.json:
            import json
            print(json.dumps(run_info, indent=2, default=str))
        elif args.artifacts_only:
            print(f"\n{'='*60}")
            print(f"Artifacts for Run: {run_info['run_name']} ({args.run_id})")
            print(f"Experiment: {run_info['experiment_name']}")
            print(f"{'='*60}")
            print(f"Artifact URI: {run_info['artifact_uri']}")
            
            if run_info['artifacts']:
                print(f"\nArtifacts ({len(run_info['artifacts'])}):")
                for artifact in run_info['artifacts']:
                    if artifact['is_dir']:
                        print(f"  📁 {artifact['path']}/")
                        if 'contents' in artifact and artifact['contents']:
                            for sub_artifact in artifact['contents']:
                                size_str = f" ({sub_artifact['file_size']} bytes)" if sub_artifact['file_size'] else ""
                                icon = "📁" if sub_artifact['is_dir'] else "📄"
                                print(f"    {icon} {sub_artifact['path']}{size_str}")
                    else:
                        size_str = f" ({artifact['file_size']} bytes)" if artifact['file_size'] else ""
                        print(f"  📄 {artifact['path']}{size_str}")
            else:
                print("\nNo artifacts found for this run.")
        else:
            print_run_info(run_info, args.verbose)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 