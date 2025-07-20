#!/usr/bin/env python3
"""
MLflow Experiment Grid Creator

This script queries MLflow for recent clustering and probing experiments,
extracts their artifact images, and creates grid visualizations (4x5) with
metadata below each image.

Usage:
    python create_experiment_grids.py [--days-back 7] [--output-dir images/grids]
"""

import mlflow
import mlflow.tracking
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import shutil
import re


def setup_mlflow_tracking(tracking_uri: Optional[str] = None) -> None:
    """Setup MLflow tracking URI."""
    if tracking_uri is None:
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


def get_recent_experiments(days_back: int = 7, experiment_type: str = "clustering") -> List[Dict[str, Any]]:
    """
    Get recent experiments of specified type from MLflow.
    
    Args:
        days_back: Number of days to look back
        experiment_type: Type of experiment ("clustering" or "probing")
        
    Returns:
        List of experiment dictionaries with metadata
    """
    client = mlflow.tracking.MlflowClient()
    
    # Calculate cutoff time
    cutoff_time = datetime.now() - timedelta(days=days_back)
    cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
    
    experiments = []
    
    # Get all experiments
    all_experiments = client.search_experiments()
    
    for experiment in all_experiments:
        # Filter by experiment type
        exp_name = experiment.name.lower()
        if experiment_type == "clustering":
            if not any(keyword in exp_name for keyword in ["kmeans", "cluster", "clustering"]):
                continue
        elif experiment_type == "probing":
            if not any(keyword in exp_name for keyword in ["probe", "probing", "decision_tree"]):
                continue
        else:
            continue
            
        # Get runs for this experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attribute.start_time >= {cutoff_timestamp}",
            max_results=100
        )
        
        for run in runs:
            # Extract metadata
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'experiment_name': experiment.name,
                'start_time': run.info.start_time,
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params),
                'tags': dict(run.data.tags),
                'artifact_uri': run.info.artifact_uri
            }
            
            # Extract key metrics based on experiment type
            if experiment_type == "clustering":
                # Look for purity and NMI metrics
                purity = None
                nmi = None
                for metric_name, value in run_info['metrics'].items():
                    if 'purity' in metric_name.lower():
                        purity = value
                    elif 'nmi' in metric_name.lower():
                        nmi = value
                
                run_info['purity'] = purity
                run_info['nmi'] = nmi
                
            elif experiment_type == "probing":
                # Look for accuracy
                accuracy = run_info['metrics'].get('accuracy')
                run_info['accuracy'] = accuracy
            
            # Extract layer number
            layer = None
            layer_param = run_info['params'].get('layer_num') or run_info['params'].get('layer')
            if layer_param:
                layer = int(layer_param)
            else:
                # Try to extract from run name
                layer_match = re.search(r'layer[_]?(\d+)', run_info['run_name'], re.IGNORECASE)
                if layer_match:
                    layer = int(layer_match.group(1))
            
            run_info['layer'] = layer
            
            # Check for EC configuration (prioritize)
            is_ec = False
            if 'ec' in run_info['run_name'].lower() or 'ec' in run_info['experiment_name'].lower():
                is_ec = True
            elif any('ec' in str(v).lower() for v in run_info['params'].values()):
                is_ec = True
            
            run_info['is_ec'] = is_ec
            
            experiments.append(run_info)
    
    return experiments


def get_artifact_image(run_id: str, client: mlflow.tracking.MlflowClient) -> Optional[str]:
    """
    Get the path to an image artifact for a run.
    
    Args:
        run_id: MLflow run ID
        client: MLflow client
        
    Returns:
        Local path to downloaded image or None if not found
    """
    try:
        artifacts = client.list_artifacts(run_id)
        
        for artifact in artifacts:
            if artifact.path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Download the artifact
                local_path = client.download_artifacts(run_id, artifact.path)
                return local_path
                
        # Check subdirectories
        for artifact in artifacts:
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_artifact in sub_artifacts:
                    if sub_artifact.path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        local_path = client.download_artifacts(run_id, sub_artifact.path)
                        return local_path
                        
    except Exception as e:
        print(f"Error getting artifact for {run_id}: {e}")
        
    return None


def filter_and_sample_experiments(experiments: List[Dict[str, Any]], 
                                 dataset: str = "snli", 
                                 experiment_type: str = "clustering",
                                 target_count: int = 20) -> List[Dict[str, Any]]:
    """
    Filter experiments by dataset and sample the best ones.
    
    Args:
        experiments: List of experiment dictionaries
        dataset: Target dataset ("snli" or "folio")
        experiment_type: Type of experiment
        target_count: Number of experiments to return
        
    Returns:
        Filtered and sampled experiments
    """
    # Filter by dataset
    filtered = []
    for exp in experiments:
        exp_name = exp['experiment_name'].lower()
        run_name = exp['run_name'].lower()
        
        if dataset.lower() in exp_name or dataset.lower() in run_name:
            filtered.append(exp)
    
    # Prioritize EC experiments
    ec_experiments = [exp for exp in filtered if exp['is_ec']]
    non_ec_experiments = [exp for exp in filtered if not exp['is_ec']]
    
    # Sort by relevant metric
    if experiment_type == "clustering":
        ec_experiments.sort(key=lambda x: x['purity'] or 0, reverse=True)
        non_ec_experiments.sort(key=lambda x: x['purity'] or 0, reverse=True)
    elif experiment_type == "probing":
        ec_experiments.sort(key=lambda x: x['accuracy'] or 0, reverse=True)
        non_ec_experiments.sort(key=lambda x: x['accuracy'] or 0, reverse=True)
    
    # Sample experiments (prioritize EC)
    sampled = []
    sampled.extend(ec_experiments[:min(target_count, len(ec_experiments))])
    
    remaining = target_count - len(sampled)
    if remaining > 0:
        sampled.extend(non_ec_experiments[:remaining])
    
    return sampled[:target_count]


def create_experiment_grid(experiments: List[Dict[str, Any]], 
                          experiment_type: str,
                          dataset: str,
                          output_path: str,
                          grid_size: Tuple[int, int] = (4, 5)) -> bool:
    """
    Create a grid visualization of experiment images.
    
    Args:
        experiments: List of experiment dictionaries
        experiment_type: Type of experiment
        dataset: Dataset name
        output_path: Output file path
        grid_size: Grid dimensions (cols, rows)
        
    Returns:
        True if successful, False otherwise
    """
    client = mlflow.tracking.MlflowClient()
    cols, rows = grid_size
    total_slots = cols * rows
    
    # Ensure we don't exceed grid size
    experiments = experiments[:total_slots]
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(16, 20))
    fig.suptitle(f'{experiment_type.title()} Experiments - {dataset.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    flat_axes = [ax for row in axes for ax in row] if rows > 1 else axes
    
    successful_plots = 0
    
    for idx, exp in enumerate(experiments):
        if idx >= total_slots:
            break
            
        ax = flat_axes[idx]
        
        # Get image artifact
        image_path = get_artifact_image(exp['run_id'], client)
        
        if image_path and os.path.exists(image_path):
            try:
                # Load and display image
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.axis('off')
                
                # Create metadata text
                layer = exp['layer'] or 'N/A'
                
                if experiment_type == "clustering":
                    purity = f"{exp['purity']:.4f}" if exp['purity'] else 'N/A'
                    nmi = f"{exp['nmi']:.6f}" if exp['nmi'] else 'N/A'
                    text = f"Layer: {layer}\nPurity: {purity}\nNMI: {nmi}"
                elif experiment_type == "probing":
                    accuracy = f"{exp['accuracy']:.4f}" if exp['accuracy'] else 'N/A'
                    text = f"Layer: {layer}\nAccuracy: {accuracy}"
                else:
                    text = f"Layer: {layer}"
                
                # Add text below image
                ax.text(0.5, -0.15, text, transform=ax.transAxes, 
                       ha='center', va='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
                
                # Add EC indicator if applicable
                if exp['is_ec']:
                    ax.text(0.05, 0.95, 'EC', transform=ax.transAxes,
                           ha='left', va='top', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8),
                           color='white')
                
                successful_plots += 1
                
            except Exception as e:
                print(f"Error processing image for {exp['run_id']}: {e}")
                ax.text(0.5, 0.5, 'Image\nError', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            # No image found
            ax.text(0.5, 0.5, 'No Image\nFound', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(experiments), total_slots):
        flat_axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created grid with {successful_plots} images: {output_path}")
    return successful_plots > 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create experiment grid visualizations from MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Number of days to look back for experiments (default: 7)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports_src/images/grids',
        help='Output directory for grid images (default: reports_src/images/grids)'
    )
    
    parser.add_argument(
        '--tracking-uri',
        help='MLflow tracking URI (default: auto-detect)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_mlflow_tracking(args.tracking_uri)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for experiments from the last {args.days_back} days...")
    
    # Process clustering experiments
    print("\n" + "="*60)
    print("CLUSTERING EXPERIMENTS")
    print("="*60)
    
    clustering_experiments = get_recent_experiments(args.days_back, "clustering")
    print(f"Found {len(clustering_experiments)} clustering experiments")
    
    # SNLI clustering
    snli_clustering = filter_and_sample_experiments(
        clustering_experiments, "snli", "clustering", 20
    )
    print(f"Selected {len(snli_clustering)} SNLI clustering experiments")
    
    if snli_clustering:
        success = create_experiment_grid(
            snli_clustering, "clustering", "snli",
            output_dir / "snli_clustering_grid.png"
        )
        if not success:
            print("Warning: Failed to create SNLI clustering grid")
    
    # FOLIO clustering
    folio_clustering = filter_and_sample_experiments(
        clustering_experiments, "folio", "clustering", 20
    )
    print(f"Selected {len(folio_clustering)} FOLIO clustering experiments")
    
    if folio_clustering:
        success = create_experiment_grid(
            folio_clustering, "clustering", "folio",
            output_dir / "folio_clustering_grid.png"
        )
        if not success:
            print("Warning: Failed to create FOLIO clustering grid")
    
    # Process probing experiments
    print("\n" + "="*60)
    print("PROBING EXPERIMENTS")
    print("="*60)
    
    probing_experiments = get_recent_experiments(args.days_back, "probing")
    print(f"Found {len(probing_experiments)} probing experiments")
    
    # SNLI probing
    snli_probing = filter_and_sample_experiments(
        probing_experiments, "snli", "probing", 20
    )
    print(f"Selected {len(snli_probing)} SNLI probing experiments")
    
    if snli_probing:
        success = create_experiment_grid(
            snli_probing, "probing", "snli",
            output_dir / "snli_probing_grid.png"
        )
        if not success:
            print("Warning: Failed to create SNLI probing grid")
    
    # FOLIO probing
    folio_probing = filter_and_sample_experiments(
        probing_experiments, "folio", "probing", 20
    )
    print(f"Selected {len(folio_probing)} FOLIO probing experiments")
    
    if folio_probing:
        success = create_experiment_grid(
            folio_probing, "probing", "folio",
            output_dir / "folio_probing_grid.png"
        )
        if not success:
            print("Warning: Failed to create FOLIO probing grid")
    
    print(f"\nGrid images saved to: {output_dir}")


if __name__ == "__main__":
    main() 